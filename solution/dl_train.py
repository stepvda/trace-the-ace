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
                      max_len=3072, batch=16, epochs=3, lr=2e-5,
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

    # SESSION-grouped hold-out. Siblings (same session, different objective) share an
    # IDENTICAL transcript, so an objective-only split leaks them across train/val and
    # inflates the val AUROC that drives the ensemble weight/gate. Group by session so
    # no sibling text is seen in both. (Falls back to objective grouping if no session_id.)
    rng = np.random.RandomState(42)
    grp_col = "session_id" if "session_id" in train_df.columns else "learning_objective_id"
    grp = train_df[grp_col].astype(str).values
    uniq = np.unique(grp); rng.shuffle(uniq)
    n_val_g = max(1, int(0.15 * len(uniq)))
    val_g = set(uniq[:n_val_g])
    is_val = np.array([g in val_g for g in grp])
    tr_df = train_df[~is_val].reset_index(drop=True)
    va_df = train_df[is_val].reset_index(drop=True)
    log(f"[dl] train={len(tr_df)} val={len(va_df)} ({grp_col}-grouped hold-out)")

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

    model = AutoModelForSequenceClassification.from_pretrained(base_dir, num_labels=2)
    model.to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(tr_df) / batch))
    total_steps = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    tr_loader = DataLoader(DS(tr_df.text.values, tr_df.y.values), batch_size=batch,
                           shuffle=True, collate_fn=collate)
    step = 0; stop = False
    for ep in range(epochs):
        for enc, yb in tr_loader:
            with _amp_ctx(use_amp):
                out = model(**enc); loss = torch.nn.functional.cross_entropy(out.logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                log(f"[dl] ep{ep} step{step}/{total_steps} loss={loss.item():.4f} elapsed={int(time.time()-t_start)}s")
            if time.time() - t_start > time_budget_s:
                log(f"[dl] time budget hit at step {step}"); stop = True; break
        if stop:
            break

    @torch.no_grad()
    def predict(texts):
        model.eval(); probs = []
        for i in range(0, len(texts), 64):
            chunk = list(texts[i:i + 64])
            enc = tok(chunk, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with _amp_ctx(use_amp):
                logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(len(texts))

    val_prob = predict(va_df.text.values) if len(va_df) else np.array([])
    test_prob = predict(test_texts)
    return {
        "test_prob": test_prob, "val_prob": val_prob,
        "val_y": va_df.y.values if len(va_df) else np.array([]),
        "val_ids": va_df.response_id.values if len(va_df) else np.array([]),
        "ok": True, "info": f"device={device} steps={step} secs={int(time.time()-t_start)}",
    }
