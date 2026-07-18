"""Domain-adaptive pretraining (DAPT) of ModernBERT-base on the tutoring-transcript corpus,
on the RunPod GPU. Continued MLM (ModernBERT uses 30% masking) over the role-tagged dialogues,
then save the adapted encoder for fine-tuning. Fable Session-2 step: proven, transferable,
zero container-time cost (weights bundle). ~1 epoch @3072.

Same recipe as gpu_mbert: flash_attention_2 (no sdpa-NaN), fp32 master weights + autocast(bf16).
Usage: python gpu_dapt.py <out_dir> <max_len> <epochs> <block_tokens>
"""
import os, sys, time, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
import dl_common as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("MBERT_DIR", "/workspace/models/ModernBERT-base")
TDIR = os.path.join(ROOT, "data", "train_transcripts")


def build_corpus():
    """One role-tagged dialogue document per UNIQUE session (dedup — the transcript is the domain
    signal; the objective is not part of DAPT). Uses the same T+/T- tagging as fine-tuning so the
    adapted model sees the shipped vocabulary."""
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    seen, docs = set(), []
    for sid in f["session_id"].astype(str):
        if sid in seen:
            continue
        seen.add(sid)
        p = os.path.join(TDIR, f"{sid}.csv")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:
            continue
        turns = D._parse_turns(df)
        lines = D._tagged_lines(turns, proxy_tags=True)
        if lines:
            docs.append(" ".join(lines))
    return docs


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/ModernBERT-dapt"
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    block = int(sys.argv[4]) if len(sys.argv) > 4 else max_len
    lr = 5e-5
    device = "cuda"
    t0 = time.time()
    print(f"=== DAPT ModernBERT max_len={max_len} epochs={epochs} block={block} ===", flush=True)

    docs = build_corpus()
    print(f"corpus: {len(docs)} unique-session documents ({int(time.time()-t0)}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)

    # tokenize + pack into fixed blocks (standard MLM chunking)
    ids_all = []
    for i, d in enumerate(docs):
        ids_all.extend(tok(d, add_special_tokens=False)["input_ids"] + [tok.sep_token_id])
        if i % 3000 == 0:
            print(f"  tokenized {i}/{len(docs)}", flush=True)
    n_blocks = len(ids_all) // block
    blocks = [ids_all[i * block:(i + 1) * block] for i in range(n_blocks)]
    print(f"packed {n_blocks} blocks of {block} tokens ({int(time.time()-t0)}s)", flush=True)

    class BlocksDS(Dataset):
        def __init__(self, blocks): self.b = blocks
        def __len__(self): return len(self.b)
        def __getitem__(self, i): return {"input_ids": self.b[i]}

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.30)
    loader = DataLoader(BlocksDS(blocks), batch_size=8, shuffle=True, collate_fn=collator)

    m = AutoModelForMaskedLM.from_pretrained(BASE, reference_compile=False,
                                             attn_implementation="flash_attention_2").to(device)
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    torch.manual_seed(0)
    m.train()
    step = 0
    for ep in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step{step}/{total} loss={float(loss):.4f} ({int(time.time()-t0)}s)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print(f"DAPT_DONE saved to {out_dir} ({int(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    main()
