"""Session-2 pre-flight: verify fold-matching + the DAPT MLM path run without error before the
long run. No long training — just structural + finite-loss checks."""
import os, sys
sys.path.insert(0, "/workspace/trace/solution")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import dl_common as D

ROOT = "/workspace/trace"; BASE = "/workspace/models/ModernBERT-base"
f = pd.read_csv(ROOT + "/data/train_features.csv")
lab = pd.read_csv(ROOT + "/data/train_labels.csv").set_index("response_id")
f["y"] = lab.loc[f.response_id, "is_correct"].values
y = f.y.to_numpy(int); g = f.learning_objective_id.astype(str).to_numpy()
folds = np.full(len(f), -1)
for k, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(f, y, g)):
    folds[va] = k
clean = all(len(set(folds[g == gg])) == 1 for gg in np.unique(g))
print("PREFLIGHT folds sizes=%s no-objective-spans-2-folds=%s" % (np.bincount(folds).tolist(), clean), flush=True)

# text builders sanity
D.HISTORY_WORDS = 0
c = D.build_texts(f.head(50), ROOT + "/data/train_transcripts", n_words=1600, centered=True, proxy_tags=True)
D.HISTORY_WORDS = 400
h = D.build_texts(f.head(50), ROOT + "/data/train_transcripts", n_words=1600, centered=True, proxy_tags=True)
print("PREFLIGHT control_has_relevant=%.2f history_has_History=%.2f" % (
    np.mean(["Relevant:" in t for t in c]), np.mean(["History:" in t for t in h])), flush=True)

# DAPT MLM forward/backward finite (the genuinely new code path)
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForMaskedLM.from_pretrained(BASE, reference_compile=False,
                                         attn_implementation="flash_attention_2").cuda()
m.config.use_cache = False
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.30)
ids = tok("hello there let us practice adding fractions with common denominators today okay",
          add_special_tokens=False)["input_ids"]
blocks = [{"input_ids": (ids * 30)[:512]} for _ in range(4)]
batch = coll(blocks); batch = {k: v.cuda() for k, v in batch.items()}
opt = torch.optim.AdamW(m.parameters(), lr=5e-5); m.train()
for s in range(2):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(**batch).loss
    loss.backward(); opt.step(); opt.zero_grad()
    print("PREFLIGHT MLM step%d loss=%.4f finite=%s" % (s, float(loss), np.isfinite(float(loss))), flush=True)
print("PREFLIGHT_OK", flush=True)
