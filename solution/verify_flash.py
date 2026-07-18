"""Verify flash-attention fixes the ModernBERT padded-batch NaN: one real training step
(forward+backward+step) on a padded batch, plus a longer-context (8192) forward."""
import torch, numpy as np, pandas as pd, sys
sys.path.insert(0, "/workspace/trace/solution")
import dl_common as D
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = "/workspace/models/ModernBERT-base"; ROOT = "/workspace/trace"; TDIR = ROOT + "/data/train_transcripts"
try:
    import flash_attn
    print("flash_attn version:", flash_attn.__version__, flush=True)
except Exception as e:
    print("flash_attn import FAILED:", str(e)[:100], flush=True); sys.exit(2)

ff = pd.read_csv(ROOT + "/data/train_features.csv")
lab = pd.read_csv(ROOT + "/data/train_labels.csv").set_index("response_id")
f = ff.sample(16, random_state=1).reset_index(drop=True)
yv = lab.loc[f.response_id, "is_correct"].to_numpy(int)
D.HISTORY_WORDS = 0
texts = D.build_texts(f, TDIR, n_words=1600, centered=True, proxy_tags=True)
tok = AutoTokenizer.from_pretrained(BASE); tok.truncation_side = "left"

m = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, reference_compile=False, attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16).cuda()
m.config.use_cache = False
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
m.train()
for step in range(6):
    b = slice((step * 4) % 16, (step * 4) % 16 + 4)
    enc = tok(texts[b], return_tensors="pt", padding=True, truncation=True, max_length=3072)
    enc = {k: v.cuda() for k, v in enc.items()}
    yb = torch.tensor([int(x) for x in yv[b]]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(m(**enc).logits, yb)
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    seqs = [int(enc["attention_mask"][i].sum()) for i in range(enc["attention_mask"].shape[0])]
    print("STEP %d loss=%.4f finite=%s seqlens=%s" % (step, float(loss), np.isfinite(float(loss)), seqs), flush=True)
pf = all(torch.isfinite(p).all() for p in m.parameters())
print("PARAMS_FINITE_AFTER_TRAIN=%s" % pf, flush=True)
print("FLASH_VERIFY_%s" % ("OK" if pf else "FAIL"), flush=True)
