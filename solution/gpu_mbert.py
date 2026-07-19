"""Real-ModernBERT A/B on the RunPod 4090 — the FAITHFUL instrument (replaces the DistilBERT
proxy, which was a local-memory workaround). Three arms on ModernBERT-base, objective-grouped
holdout (all val objectives unseen — matches the test regime), single seed for screening:

  control : shipped centered texts (History off), max_len 3072  -> instrument calibration + timing
  full    : FULL transcript, on-objective turns highlighted "* ", max_len 8192  -> the big unlock
  history : centered + additive History field (~400w) at 3072   -> cheap fallback if 8192 busts budget

Reports held-out-objective AUROC + wall-clock per arm (to project the A100 6h budget:
4090 bf16 ~= 0.55x A100 throughput; scale by 35000/subset for full data). Mirrors the container
trainer: bf16 autocast, LEFT truncation, best-checkpoint-on-val-AUROC, reference_compile=False (P0),
attn_implementation pinned to sdpa (container has no flash-attn).

Usage: python gpu_mbert.py <arms csv> <subset> <epochs> <seed>
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
import dl_common as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("MBERT_DIR", "/workspace/models/ModernBERT-base")
TDIR = os.path.join(ROOT, "data", "train_transcripts")

ARMS = {  # (max_len, batch, accum, builder kwargs, full_context) — effective batch 16 each.
    # ModernBERT global-attention layers are memory-heavy at long seq; with gradient
    # checkpointing these fit the 24GB 4090 without OOM (OOM recovery corrupts CUDA state).
    "control": dict(max_len=3072, batch=8, accum=2, full=False,
                    kw=dict(HISTORY_WORDS=0,   RELEVANT_WORDS=600, RECENT_WORDS=1000)),
    "full":    dict(max_len=8192, batch=4, accum=4, full=True,  kw=dict()),
    "history": dict(max_len=3072, batch=8, accum=2, full=False,
                    kw=dict(HISTORY_WORDS=400, RELEVANT_WORDS=600, RECENT_WORDS=1000)),
    # ModernBERT-LARGE probe: same 'control' focused rep, smaller batch (2.6x params) -> effective 16.
    # Run with MBERT_DIR=.../ModernBERT-large. p_ column is 'control_large' (distinct from base 'control').
    "control_large": dict(max_len=3072, batch=4, accum=4, full=False,
                          kw=dict(HISTORY_WORDS=0, RELEVANT_WORDS=600, RECENT_WORDS=1000)),
}


def build_texts_for(f, arm):
    cfg = ARMS[arm]
    for k, v in cfg["kw"].items():
        setattr(D, k, v)
    if cfg["full"]:
        return D.build_texts(f, TDIR, full_context=True, proxy_tags=True)
    return D.build_texts(f, TDIR, n_words=1600, centered=True, proxy_tags=True)


def load_model():
    # flash_attention_2 REQUIRED: ModernBERT's sdpa path NaNs on padded batches. flash-attn
    # unpads, so it is both the correctness fix and memory-efficient (needed for 8192). We
    # deliberately do NOT fall back to sdpa — a silent sdpa fallback is the NaN trap that killed
    # the container transformer.
    # fp32 MASTER WEIGHTS (no torch_dtype): the training loop's autocast(bf16) gives flash-attn
    # its half-precision q/k/v, while AdamW keeps fp32 params/moments. Loading bf16 params put the
    # optimizer state in bf16 -> small updates rounded to zero -> undertrained (~0.53 vs ~0.60).
    return AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, reference_compile=False,
        attn_implementation="flash_attention_2")


class DS(Dataset):
    def __init__(self, texts, ys): self.texts = list(texts); self.ys = ys
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], int(self.ys[i])


def run_arm(arm, texts, y, tr, va, epochs, seed, batch, accum, log):
    cfg = ARMS[arm]; max_len = cfg["max_len"]
    t0 = time.time()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE); tok.truncation_side = "left"

    def collate(items):
        txt = [a for a, _ in items]; ys = [b for _, b in items]
        enc = tok(txt, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}, torch.tensor(ys, dtype=torch.long, device=device)

    tr_texts = [texts[i] for i in tr]; tr_y = y[tr]
    va_texts = [texts[i] for i in va]; va_y = y[va].astype(float)
    torch.manual_seed(seed)
    m = load_model().to(device)
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    steps = math.ceil(len(tr) / (batch * accum)) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loader = DataLoader(DS(tr_texts, tr_y), batch_size=batch, shuffle=True, collate_fn=collate)

    @torch.no_grad()
    def predict(texts_):
        m.eval(); out = []
        for i in range(0, len(texts_), 64):
            enc = tok(texts_[i:i + 64], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        for bi, (enc, yb) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(m(**enc).logits, yb) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        vp = predict(va_texts); au = float(roc_auc_score(va_y, vp))
        log(f"  [{arm}] ep{ep} val_auc={au:.4f} ({int(time.time()-t0)}s)")
        if au > best_auc:
            best_auc = au; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    if best_state:
        m.load_state_dict(best_state)
    vp = predict(va_texts)
    if not np.isfinite(vp).all():
        raise ValueError(f"{arm}: {int((~np.isfinite(vp)).sum())} non-finite predictions "
                         f"(divergence or corrupted CUDA state — check batch/lr)")
    auc = float(roc_auc_score(va_y, vp)); ll = float(log_loss(va_y, np.clip(vp, 1e-6, 1 - 1e-6)))
    dt = time.time() - t0
    del m; torch.cuda.empty_cache()
    return dict(arm=arm, auc=auc, ll=ll, best_ep_auc=best_auc, time=dt, max_len=max_len, batch=batch, accum=accum)


def main():
    arms = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["control", "full", "history"])
    subset = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    log = lambda m: print(m, flush=True)
    log(f"=== ModernBERT A/B arms={arms} subset={subset} epochs={epochs} seed={seed} "
        f"gpu={torch.cuda.get_device_name(0)} ===")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    l = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = l.loc[f.response_id, "is_correct"].values
    f = f.sample(n=subset, random_state=seed).reset_index(drop=True)
    y = f.y.to_numpy(int); groups = f.learning_objective_id.astype(str).to_numpy()
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(f, y, groups))
    log(f"subset={len(f)} objectives={len(set(groups))} train={len(tr)} val={len(va)} (obj-grouped; all val unseen)")

    res = {}
    for arm in arms:
        texts = build_texts_for(f, arm)
        salient = (float(np.mean(["* " in t for t in texts])) if ARMS[arm]["full"]
                   else float(np.mean(["History:" in t for t in texts])))
        wc = int(np.median([len(t.split()) for t in texts]))
        log(f"[{arm}] built texts median_words={wc} salient_frac={salient:.2f} max_len={ARMS[arm]['max_len']}")
        b, a = ARMS[arm]["batch"], ARMS[arm]["accum"]
        r = None
        for attempt in range(3):
            try:
                r = run_arm(arm, texts, y, tr, va, epochs, seed, b, a, log); break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); b = max(2, b // 2); a = a * 2
                    log(f"  [{arm}] CUDA OOM -> retry batch={b} accum={a}")
                else:
                    raise
        r["words"] = wc
        res[arm] = r
        log(f"=== ARM {arm}: AUROC={r['auc']:.4f} logloss={r['ll']:.5f} best_ep_auc={r['best_ep_auc']:.4f} "
            f"words={wc} time={r['time']:.0f}s (batch={r['batch']}x{r['accum']}) ===")

    log("\n==== SUMMARY (held-out-objective AUROC) ====")
    for arm in arms:
        log(f"  {arm:8s} AUROC={res[arm]['auc']:.4f} ll={res[arm]['ll']:.5f} "
            f"time={res[arm]['time']:.0f}s maxlen={res[arm]['max_len']} words={res[arm]['words']}")
    if "control" in res:
        for arm in arms:
            if arm != "control":
                d = res[arm]['auc'] - res['control']['auc']
                a100 = res[arm]['time'] / 0.55 * (35000 / subset)  # 4090->A100 + full-data scale
                log(f"  {arm}-control: AUROC {d:+.4f} | proj. A100 full-data train ~{a100/60:.1f} min "
                    f"(budget 6h incl classical+inference; keep >=25% margin)")
    log("DONE_MBERT_AB")


if __name__ == "__main__":
    main()
