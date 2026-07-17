"""A/B: does the objective-CENTERED text representation raise held-out AUROC vs the
plain TAIL representation? Same small encoder (local domain-adapted DistilBERT), same
objective-grouped split, same seed — only the text differs. The absolute AUROC of a tiny
model is not the point; the DELTA is. This validates Fable's biggest recommendation
(the transformer sees the wrong objective for ~24% of rows) BEFORE spending a scored slot.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MODEL = os.path.join(CACHE, "distilbert_adapted")   # local, no download


def run(texts, y, tr, va, max_len, batch, epochs, seed):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.metrics import roc_auc_score, log_loss
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.truncation_side = "left"          # match the shipped container (keep most-recent tokens)

    class DS(Dataset):
        def __init__(self, idx): self.idx = list(idx)
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]
            enc = tok(texts[j], truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0], "labels": int(y[j])}

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    args = TrainingArguments(output_dir=os.path.join(CACHE, "ab_tmp"),
        per_device_train_batch_size=batch, per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, learning_rate=1.5e-5, num_train_epochs=epochs,
        eval_strategy="no", save_strategy="no", logging_steps=250, report_to=[],
        warmup_ratio=0.1, weight_decay=0.01, seed=seed, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=DS(tr))
    trainer.train()
    pred = trainer.predict(DS(va))
    p = torch.softmax(torch.tensor(pred.predictions), -1)[:, 1].numpy()
    del model, trainer
    import gc; gc.collect()
    return float(roc_auc_score(y[va], p)), float(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6)))


def main(subset=10000, max_len=512, batch=8, epochs=2, seed=42):
    import dl_common as D
    from sklearn.model_selection import StratifiedGroupKFold
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    l = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = l.loc[f.response_id, "is_correct"].values
    f = f.sample(n=subset, random_state=seed).reset_index(drop=True)
    tdir = os.path.join(ROOT, "data", "train_transcripts")
    y = f.y.to_numpy(int); groups = f.learning_objective_id.astype(str).to_numpy()
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(f, y, groups))
    print(f"subset={len(f)} objectives={len(set(groups))} train={len(tr)} val={len(va)} (obj-grouped)", flush=True)

    # reduced budgets so BOTH representations fit within max_len (fair comparison)
    D.RELEVANT_WORDS, D.RECENT_WORDS = 180, 180
    t0 = time.time()
    tail = D.build_texts(f, tdir, n_words=360, centered=False, proxy_tags=False)
    cent = D.build_texts(f, tdir, n_words=360, centered=True, proxy_tags=True)
    has_rel = float(np.mean(["Relevant:" in t for t in cent]))
    # how often does centered actually differ from tail (i.e. the fix bites)?
    differ = float(np.mean([c != t for c, t in zip(cent, tail)]))
    print(f"built texts in {time.time()-t0:.0f}s | centered-has-Relevant={has_rel:.2f} | differs-from-tail={differ:.2f}", flush=True)

    print("=== [1/2] TAIL representation (current) ===", flush=True)
    a_t, ll_t = run(tail, y, tr, va, max_len, batch, epochs, seed)
    print(f"  TAIL     AUC={a_t:.4f}  logloss={ll_t:.5f}", flush=True)
    print("=== [2/2] OBJECTIVE-CENTERED representation (new) ===", flush=True)
    a_c, ll_c = run(cent, y, tr, va, max_len, batch, epochs, seed)
    print(f"  CENTERED AUC={a_c:.4f}  logloss={ll_c:.5f}", flush=True)

    print(f"\n=== RESULT: centered − tail = AUC {a_c-a_t:+.4f}, logloss {ll_t-ll_c:+.5f} ===", flush=True)
    print("  => CENTERED HELPS (ship it)" if a_c > a_t + 0.004 else
          ("  => marginal" if a_c > a_t else "  => no gain / hurts"), flush=True)


if __name__ == "__main__":
    main()
