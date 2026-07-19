"""5-fold objective-grouped OOF for the transformer legs (control + history), on the RunPod GPU.
Reproduces the classical OOF folds EXACTLY (StratifiedGroupKFold(5, shuffle, random_state=42) on
learning_objective_id, train_features/row_ids order) so the transformer OOF aligns row-for-row with
submission/assets/classical_oof.parquet -> clean blend-weight fit.

Fixed epoch count (no best-checkpoint on the OOF val = no leakage); CV-optimal from Session 1 was
epoch 2. One seed per (rep, fold). Saves response_id, y, fold, p_<rep> for each rep.
Set MBERT_DIR to the DAPT'd encoder to run OOF on the adapted base.
Usage: python gpu_oof.py <reps csv> <epochs> <out_parquet>
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from gpu_mbert import load_model, build_texts_for, ARMS, DS, ROOT, BASE


def train_predict(rep, texts, y, tr, va, max_len, batch, accum, epochs, seed):
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE); tok.truncation_side = "left"

    def collate(items):
        txt = [a for a, _ in items]; ys = [b for _, b in items]
        enc = tok(txt, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}, torch.tensor(ys, dtype=torch.long, device=device)

    tr_texts = [texts[i] for i in tr]; tr_y = y[tr]
    va_texts = [texts[i] for i in va]
    torch.manual_seed(seed)
    m = load_model().to(device); m.config.use_cache = False
    if os.environ.get("RESET_HEAD"):
        # contrastive-init: MBERT_DIR points at the matching model. Reset ONLY the final decision
        # layer so we measure the pretrained BACKBONE, not a transferred match/no-match boundary.
        import torch.nn as nn
        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Linear):
            m.classifier.reset_parameters()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    steps = math.ceil(len(tr) / (batch * accum)) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loader = DataLoader(DS(tr_texts, tr_y), batch_size=batch, shuffle=True, collate_fn=collate)
    m.train()
    for ep in range(epochs):
        opt.zero_grad()
        for bi, (enc, yb) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(m(**enc).logits, yb) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()

    @torch.no_grad()
    def predict(texts_):
        m.eval(); out = []
        for i in range(0, len(texts_), 64):
            enc = tok(texts_[i:i + 64], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    vp = predict(va_texts)
    del m; torch.cuda.empty_cache()
    return vp


def main():
    reps = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["control", "history"])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = sys.argv[3] if len(sys.argv) > 3 else "/workspace/oof_transformer.parquet"
    seed_base = int(sys.argv[4]) if len(sys.argv) > 4 else 42   # per-fold seed = seed_base + k
    print(f"=== OOF reps={reps} epochs={epochs} seed_base={seed_base} base={BASE} ===", flush=True)
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = lab.loc[f.response_id, "is_correct"].values
    y = f.y.to_numpy(int); groups = f.learning_objective_id.astype(str).to_numpy()
    rid = f.response_id.astype(str).to_numpy()
    t0 = time.time()
    texts = {rep: build_texts_for(f, rep) for rep in reps}
    print(f"built texts for {reps} ({int(time.time()-t0)}s)", flush=True)

    oof = {rep: np.zeros(len(f)) for rep in reps}
    fold_arr = np.full(len(f), -1)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)  # SAME as gen_classical_oof
    for k, (tr, va) in enumerate(sgkf.split(f, y, groups)):
        fold_arr[va] = k
        for rep in reps:
            cfg = ARMS[rep]; ts = time.time()
            vp = train_predict(rep, texts[rep], y, tr, va, cfg["max_len"], cfg["batch"],
                               cfg["accum"], epochs, seed=seed_base + k)
            oof[rep][va] = vp
            print(f"[fold{k} {rep}] val_auc={roc_auc_score(y[va], vp):.4f} n={len(va)} "
                  f"({int(time.time()-ts)}s, tot {int(time.time()-t0)}s)", flush=True)

    df = pd.DataFrame({"response_id": rid, "y": y, "fold": fold_arr})
    for rep in reps:
        df["p_" + rep] = oof[rep]
        print(f"=== OOF {rep}: AUROC={roc_auc_score(y, oof[rep]):.4f} "
              f"logloss={log_loss(y, np.clip(oof[rep], 1e-6, 1 - 1e-6)):.5f} ===", flush=True)
    df.to_parquet(out)
    print(f"OOF_DONE saved {out}", flush=True)


if __name__ == "__main__":
    main()
