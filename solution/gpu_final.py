"""Final bundle trainer (RunPod GPU): train N seeds per representation on ALL 35k train rows
(no holdout — the OOF already validated the design and picked epochs), save each as an fp16
ModernBERT checkpoint for the inference-only container to bundle + ensemble.

Set MBERT_DIR to the (possibly DAPT'd) base chosen in Session 2a. Epochs = the CV-optimal from
the OOF (default 2). Reps = whatever blend_gate.py selected (e.g. "control,history").
Usage: python gpu_final.py <reps csv> <seeds_per_rep> <epochs> <out_root>
Outputs: <out_root>/<rep>_seed<seed>/ (config.json, model.safetensors fp16, tokenizer files)
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gpu_mbert import load_model, build_texts_for, ARMS, DS, ROOT, BASE


def train_full(rep, texts, y, max_len, batch, accum, epochs, seed, out_dir, log):
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE); tok.truncation_side = "left"

    def collate(items):
        txt = [a for a, _ in items]; ys = [b for _, b in items]
        enc = tok(txt, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}, torch.tensor(ys, dtype=torch.long, device=device)

    torch.manual_seed(seed)
    m = load_model().to(device); m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    steps = math.ceil(len(texts) / (batch * accum)) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loader = DataLoader(DS(texts, y), batch_size=batch, shuffle=True, collate_fn=collate)
    t0 = time.time(); m.train()
    for ep in range(epochs):
        opt.zero_grad()
        for bi, (enc, yb) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(m(**enc).logits, yb) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        log(f"  [{rep} seed{seed}] ep{ep} done ({int(time.time()-t0)}s)")
    # save fp16 (halves bundle size; inference runs bf16 under autocast anyway)
    m.half()
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    del m; torch.cuda.empty_cache()
    log(f"SAVED {out_dir} ({int(time.time()-t0)}s)")


def main():
    reps = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["control", "history"])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out_root = sys.argv[4] if len(sys.argv) > 4 else "/workspace/mbert_seeds"
    log = lambda m: print(m, flush=True)
    log(f"=== FINAL bundle reps={reps} seeds={n_seeds} epochs={epochs} base={BASE} out={out_root} ===")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = lab.loc[f.response_id, "is_correct"].values
    y = f.y.to_numpy(int)
    for rep in reps:
        texts = build_texts_for(f, rep)
        cfg = ARMS[rep]
        for s in range(n_seeds):
            seed = 42 + s
            out_dir = os.path.join(out_root, f"{rep}_seed{seed}")
            train_full(rep, texts, y, cfg["max_len"], cfg["batch"], cfg["accum"], epochs, seed, out_dir, log)
    log("FINAL_DONE")


if __name__ == "__main__":
    main()
