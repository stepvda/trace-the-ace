"""Diagnose ModernBERT NaN: find an attention-impl / precision combo that gives finite loss
on a REAL padded batch (padding is what triggers the NaN — identical-length batches are fine)."""
import torch, numpy as np, pandas as pd, sys, os
sys.path.insert(0, "/workspace/trace/solution")
import dl_common as D
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = "/workspace/models/ModernBERT-base"; ROOT = "/workspace/trace"; TDIR = ROOT + "/data/train_transcripts"
ff = pd.read_csv(ROOT + "/data/train_features.csv")
lab = pd.read_csv(ROOT + "/data/train_labels.csv").set_index("response_id")
f = ff.sample(8, random_state=1).reset_index(drop=True)
yv = lab.loc[f.response_id, "is_correct"].to_numpy(int)
D.HISTORY_WORDS = 0
texts = D.build_texts(f, TDIR, n_words=1600, centered=True, proxy_tags=True)
tok = AutoTokenizer.from_pretrained(BASE); tok.truncation_side = "left"
enc4 = tok(texts[:4], return_tensors="pt", padding=True, truncation=True, max_length=3072)
print("PROBE seqlens", [int(enc4["attention_mask"][i].sum()) for i in range(4)],
      "padded_to", enc4["input_ids"].shape[1], flush=True)


def test(attn, amp):
    try:
        torch.manual_seed(0)
        m = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=2, reference_compile=False, attn_implementation=attn).cuda()
        m.train()
        enc = {k: v.cuda() for k, v in enc4.items()}
        yb = torch.tensor([int(x) for x in yv[:4]]).cuda()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cuda", enabled=False)
        with ctx:
            logits = m(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits, yb)
        print("CFG attn=%s bf16=%s logits_finite=%s loss=%.4f finite=%s"
              % (attn, amp, bool(torch.isfinite(logits).all()), float(loss), np.isfinite(float(loss))), flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print("CFG attn=%s bf16=%s ERR %s" % (attn, amp, str(e)[:80]), flush=True)
        torch.cuda.empty_cache()


for attn in ["sdpa", "eager"]:
    for amp in [True, False]:
        test(attn, amp)
