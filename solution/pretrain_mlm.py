"""Domain/task-adaptive MLM warmup: continue-pretrain DistilBERT on the tutoring
corpus (MathDial + our transcripts) with masked-language-modeling, then save the
adapted encoder so it can be fine-tuned like the base model.  torch + transformers
only. Capped by max_steps to stay feasible on the M1.
"""
import os, sys, time, math, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def main(model_name="distilbert-base-uncased", max_len=256, batch=8, max_steps=2500,
         lr=5e-5, out_dir=None):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                              DataCollatorForLanguageModeling, get_linear_schedule_with_warmup)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = out_dir or os.path.join(CACHE, "distilbert_adapted")
    print(f"device={dev} model={model_name} max_len={max_len} batch={batch} max_steps={max_steps}", flush=True)

    lines = [l.strip() for l in open(os.path.join(CACHE, "pretrain_corpus.txt")) if l.strip()]
    print(f"corpus docs={len(lines)}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, texts): self.texts = texts
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            enc = tok(self.texts[i], truncation=True, max_length=max_len)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
    loader = DataLoader(DS(lines), batch_size=batch, shuffle=True, collate_fn=collator)

    model = AutoModelForMaskedLM.from_pretrained(model_name).to(dev); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * max_steps), max_steps)

    t0 = time.time(); step = 0
    while step < max_steps:
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(**b); loss = out.loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 100 == 0:
                print(f"[mlm] step {step}/{max_steps} loss={loss.item():.4f} elapsed={int(time.time()-t0)}s", flush=True)
            if step >= max_steps:
                break
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print(f"[mlm] saved adapted encoder -> {out_dir} ({int(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max_steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    a = ap.parse_args()
    main(a.model, a.max_len, a.batch, a.max_steps)
