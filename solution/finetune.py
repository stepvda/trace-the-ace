"""Fine-tune a small transformer encoder on transcripts -> is_correct.

Designed for an 8GB M1 (MPS): small model, short recency window, small batch,
gradient checkpointing. Validates LEAKAGE-FREE with an objective-grouped split
(held-out learning objectives) so we measure real generalization, not memorization.

Input text per response: "Objective: <lo>. Dialogue: <last N words, role-tagged>".
"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def build_texts(features_df, transcripts_dir, n_words=280):
    """recency text per session; returns dict sid-> role-tagged last-N-words string."""
    out = {}
    for sid in features_df["session_id"].astype(str).unique():
        p = os.path.join(transcripts_dir, f"{sid}.csv")
        if not os.path.exists(p):
            out[sid] = ""; continue
        try:
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:
            out[sid] = ""; continue
        role = d.get("role", pd.Series([""] * len(d))).astype(str).str.lower()
        content = d.get("content", pd.Series([""] * len(d))).astype(str)
        lines = [f"{('S' if r=='student' else 'T' if r=='tutor' else 'B')}: {c}"
                 for r, c in zip(role, content)]
        words = " ".join(lines).split()
        out[sid] = " ".join(words[-n_words:])
    return out


def main(model_name="microsoft/deberta-v3-xsmall", n_words=280, max_len=320,
         batch=8, accum=2, epochs=2, subset=0, seed=42, out_tag="ft"):
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from sklearn.metrics import log_loss, roc_auc_score

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev} model={model_name} max_len={max_len} batch={batch} accum={accum}", flush=True)

    feats = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    labs = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    feats["y"] = labs.loc[feats.response_id, "is_correct"].values
    if subset:
        feats = feats.sample(n=subset, random_state=seed).reset_index(drop=True)
    tdir = os.path.join(ROOT, "data", "train_transcripts")
    t0 = time.time()
    rec = build_texts(feats, tdir, n_words=n_words)
    feats["text"] = ["Objective: " + str(lo) + ". Dialogue: " + rec[str(s)]
                     for lo, s in zip(feats.learning_objective, feats.session_id)]
    print(f"built texts in {time.time()-t0:.1f}s", flush=True)

    # objective-grouped split (held-out objectives = pessimistic / realistic)
    from sklearn.model_selection import StratifiedGroupKFold
    groups = feats.learning_objective_id.astype(str).values
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(feats, feats.y, groups))
    print(f"train={len(tr_idx)} val={len(va_idx)} (val objectives held out)", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, df):
            self.texts = df["text"].tolist(); self.y = df["y"].astype(int).tolist()
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            enc = tok(self.texts[i], truncation=True, max_length=max_len, padding="max_length",
                      return_tensors="pt")
            return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                    "labels": self.y[i]}

    tr_ds = DS(feats.iloc[tr_idx]); va_ds = DS(feats.iloc[va_idx])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def metrics(evalpred):
        logits, labels = evalpred
        p = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
        return {"logloss": log_loss(labels, np.clip(p, 1e-6, 1 - 1e-6)),
                "auc": roc_auc_score(labels, p)}

    args = TrainingArguments(
        output_dir=os.path.join(CACHE, f"ft_{out_tag}"),
        per_device_train_batch_size=batch, per_device_eval_batch_size=16,
        gradient_accumulation_steps=accum, learning_rate=1.5e-5, num_train_epochs=epochs,
        eval_strategy="epoch", save_strategy="no", logging_steps=100,
        report_to=[], fp16=False, bf16=False, dataloader_num_workers=0,
        warmup_ratio=0.1, weight_decay=0.01, seed=seed, max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                      compute_metrics=metrics)
    t0 = time.time()
    trainer.train()
    print(f"train done in {time.time()-t0:.1f}s", flush=True)
    ev = trainer.evaluate()
    print("VAL", json.dumps({k: float(v) for k, v in ev.items() if isinstance(v, (int, float))}), flush=True)
    # save val OOF preds for ensembling
    pred = trainer.predict(va_ds)
    p = torch.softmax(torch.tensor(pred.predictions), -1)[:, 1].numpy()
    pd.DataFrame({"response_id": feats.iloc[va_idx].response_id.values, "p_ft": p,
                  "y": feats.iloc[va_idx].y.values}).to_csv(
        os.path.join(CACHE, f"ft_val_{out_tag}.csv"), index=False)
    print(f"AUC={roc_auc_score(feats.iloc[va_idx].y, p):.4f} "
          f"logloss={log_loss(feats.iloc[va_idx].y, np.clip(p,1e-6,1-1e-6)):.5f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/electra-small-discriminator")
    ap.add_argument("--n_words", type=int, default=280)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--out_tag", default="ft")
    a = ap.parse_args()
    main(a.model, a.n_words, a.max_len, a.batch, a.accum, a.epochs, a.subset, out_tag=a.out_tag)
