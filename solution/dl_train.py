"""In-container fine-tune + inference for the transformer, using only torch +
transformers (no accelerate/datasets). GPU-aware (CUDA in the competition
container; CPU fallback locally). Respects a wall-clock budget and a smoke mode.

train_and_predict(base_dir, train_df, test_texts, ...) ->
    dict(test_prob, val_prob, val_ids, val_y, ok, info)

The caller (main.py) uses the held-out val performance to decide the ensemble
weight vs. the classical model, and falls back to classical if ok is False.
"""
import os, time, math
import contextlib
import numpy as np
import pandas as pd


def _amp_ctx(use_amp):
    import torch
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_and_predict(base_dir, train_df, test_texts, *, smoke=False,
                      max_len=3072, batch=16, epochs=4, lr=2e-5,
                      time_budget_s=12000, log=print):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    t_start = time.time()
    device = _pick_device()
    use_amp = (device == "cuda")
    if smoke:
        # tiny run just to validate the code path end-to-end
        train_df = train_df.sample(n=min(600, len(train_df)), random_state=0).reset_index(drop=True)
        max_len, batch, epochs = 256, 8, 1
        time_budget_s = min(time_budget_s, 300)
    if device == "cpu":
        max_len = min(max_len, 384); batch = min(batch, 8)
    log(f"[dl] device={device} n_train={len(train_df)} max_len={max_len} batch={batch} epochs={epochs} smoke={smoke}")

    # DOUBLY-DISJOINT hold-out. The test holds out OBJECTIVES, so val must too — a
    # session-only split leaves val objectives SEEN in train and rewards objective
    # memorization (which is worth zero on the real test). AND siblings (same session,
    # different objective) share an identical transcript, so we must not split a session
    # across train/val either. So: hold out 15% of OBJECTIVES, then move any session that
    # touches a held-out objective WHOLLY into val. Val is then unseen-objective AND
    # sibling-leak-free — the honest signal for the ensemble weight/gate/checkpoint.
    rng = np.random.RandomState(42)
    objs = train_df["learning_objective_id"].astype(str).values
    uniq_obj = np.unique(objs); rng.shuffle(uniq_obj)
    val_obj = set(uniq_obj[:max(1, int(0.15 * len(uniq_obj)))])
    row_valobj = np.array([o in val_obj for o in objs])
    if "session_id" in train_df.columns:
        sids = train_df["session_id"].astype(str).values
        val_sessions = set(sids[row_valobj])
        is_val = np.array([s in val_sessions for s in sids])
    else:
        is_val = row_valobj
    tr_df = train_df[~is_val].reset_index(drop=True)
    va_df = train_df[is_val].reset_index(drop=True)
    va_valobj = row_valobj[is_val]   # which val rows have a HELD-OUT objective (for gate honesty)
    log(f"[dl] train={len(tr_df)} val={len(va_df)} (doubly-disjoint: {len(val_obj)} held-out objectives + whole sessions)")

    tok = AutoTokenizer.from_pretrained(base_dir)
    # LEFT truncation: the assessment-relevant content is at the END of the session
    # (the tail is what predicts the next-question outcome). Default right-truncation
    # would cut exactly that ending off long transcripts. Keep the most-recent tokens.
    tok.truncation_side = "left"

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=max_len, padding=False)

    class DS(Dataset):
        def __init__(self, texts, ys=None):
            self.texts = list(texts); self.ys = ys
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return (self.texts[i], -1 if self.ys is None else int(self.ys[i]))

    def collate(batch_items):
        texts = [b[0] for b in batch_items]; ys = [b[1] for b in batch_items]
        enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        return enc, torch.tensor(ys, dtype=torch.long, device=device)

    # P0 (prime suspect for the id-1579 fallback): ModernBERT enables torch.compile by
    # default (reference_compile), which dies in an offline container with no writable
    # inductor/triton cache at the first forward; sdpa attention also sidesteps an
    # incompatible flash-attn in the image. Fall back through progressively-plainer loads.
    import transformers as _tf
    log(f"[dl] transformers={_tf.__version__} torch={torch.__version__} device={device}")
    def _load(d):
        for kw in ({"reference_compile": False, "attn_implementation": "sdpa"},
                   {"attn_implementation": "sdpa"}, {}):
            try:
                return AutoModelForSequenceClassification.from_pretrained(d, num_labels=2, **kw)
            except TypeError:
                continue
        return AutoModelForSequenceClassification.from_pretrained(d, num_labels=2)
    from sklearn.metrics import roc_auc_score
    va_texts = va_df.text.values
    va_y = va_df.y.values.astype(float) if len(va_df) else np.array([])

    @torch.no_grad()
    def _predict(m, texts):
        m.eval(); probs = []
        for i in range(0, len(texts), 64):
            enc = tok(list(texts[i:i + 64]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with _amp_ctx(use_amp):
                probs.append(torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(len(texts))

    def _train_one(seed, do_preflight):
        """One fine-tune with best-checkpoint-on-val-AUROC selection. Returns (val_prob,
        test_prob, best_val_auc, steps)."""
        torch.manual_seed(seed)
        m = _load(base_dir); m.to(device)
        if do_preflight:
            # catch a runtime failure HERE (logged, container falls back) instead of
            # dying silently mid-train as id-1579 did.
            try:
                _pf = tok(list(tr_df.text.values[:4]), truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt")
                _pf = {k: v.to(device) for k, v in _pf.items()}
                m.eval()
                with torch.no_grad(), _amp_ctx(use_amp):
                    m(**_pf)
                log("[dl] preflight forward OK")
            except Exception as e:
                log(f"[dl] PREFLIGHT FAILED ({type(e).__name__}: {str(e)[:180]}) — DL leg will fall back")
                raise
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        total = max(1, math.ceil(len(tr_df) / batch)) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
        loader = DataLoader(DS(tr_df.text.values, tr_df.y.values), batch_size=batch,
                            shuffle=True, collate_fn=collate)
        best_auc, best_state, step, stop = -1.0, None, 0, False
        for ep in range(epochs):
            m.train()
            for enc, yb in loader:
                with _amp_ctx(use_amp):
                    loss = torch.nn.functional.cross_entropy(m(**enc).logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if time.time() - t_start > time_budget_s:
                    log(f"[dl] time budget hit at step {step}"); stop = True; break
            # BEST-CHECKPOINT on val AUROC (near-noise fine-tunes swing 0.01-0.02/epoch;
            # shipping the last epoch is usually the most overfit).
            if len(va_texts):
                vp = _predict(m, va_texts)
                try:
                    au = float(roc_auc_score(va_y, vp))
                except Exception:
                    au = 0.5
                log(f"[dl] seed{seed} ep{ep} step{step} val_auc={au:.4f} ({int(time.time()-t_start)}s)")
                if au > best_auc:
                    best_auc = au
                    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            if stop:
                break
        if best_state is not None:
            m.load_state_dict(best_state)          # restore the best epoch before predicting
        vp = _predict(m, va_texts) if len(va_texts) else np.array([])
        tp = _predict(m, test_texts)
        del m
        return vp, tp, best_auc, step

    # Seed-average, budget-permitting: variance reduction on a near-noise task + a steadier
    # gate. A second run only if we're under half the budget after the first.
    vp1, tp1, auc1, step = _train_one(42, do_preflight=True)
    val_list, test_list, seeds = [vp1], [tp1], [42]
    if (time.time() - t_start) < 0.5 * time_budget_s:
        log("[dl] budget remains -> second seed for averaging")
        vp2, tp2, auc2, _ = _train_one(43, do_preflight=False)
        val_list.append(vp2); test_list.append(tp2); seeds.append(43)
    val_prob = np.mean(val_list, axis=0) if len(va_texts) else np.array([])
    test_prob = np.mean(test_list, axis=0)
    return {
        "test_prob": test_prob, "val_prob": val_prob,
        "val_y": va_df.y.values if len(va_df) else np.array([]),
        "val_ids": va_df.response_id.values if len(va_df) else np.array([]),
        "val_valobj": va_valobj if len(va_df) else np.array([]),  # held-out-objective rows (gate honesty)
        "ok": True, "info": f"device={device} seeds={seeds} best_val_auc={auc1:.4f} secs={int(time.time()-t_start)}",
    }
