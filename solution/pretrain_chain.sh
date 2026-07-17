#!/bin/bash
# A/B: external-data domain-adaptive pretraining (MathDial + own transcripts).
# BERT-mini (29M) to fit the 8GB M1 solo. Runs ALONE (no concurrent heavy jobs).
cd /Users/nstephane/Dev/AI_Data_Science_training/trace-the-ace
PY=.venv/bin/python
MODEL="google/bert_uncased_L-4_H-512_A-8"
ADAPTED="solution/cache/distilbert_adapted"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false

echo "=== STAGE A: MLM warmup on tutoring corpus (MathDial + transcripts) ==="
$PY -u solution/pretrain_mlm.py --model "$MODEL" --max_steps 3000 --batch 8 --max_len 256 2>&1 \
   | grep -aE "device=|corpus|\[mlm\] step (500|1000|1500|2000|2500|3000)|saved|Error|RuntimeError" || true

echo "=== STAGE B: fine-tune BASELINE (no adaptation) ==="
$PY -u solution/finetune.py --model "$MODEL" --n_words 300 --max_len 256 \
   --batch 8 --epochs 2 --subset 18000 --out_tag mini_base 2>&1 | grep -aE "train=|\"eval_auc\"|AUC=|Error|RuntimeError" | tail -4 || true

echo "=== STAGE C: fine-tune ADAPTED (MathDial+transcript MLM warmup) ==="
$PY -u solution/finetune.py --model "$ADAPTED" --n_words 300 --max_len 256 \
   --batch 8 --epochs 2 --subset 18000 --out_tag mini_adapt 2>&1 | grep -aE "train=|\"eval_auc\"|AUC=|Error|RuntimeError" | tail -4 || true

echo "=== STAGE D: A/B result (adapted vs base) ==="
$PY - <<'PY' 2>&1 || true
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss
C="solution/cache"
def m(tag):
    d=pd.read_csv(f"{C}/ft_val_{tag}.csv"); y=d.y.values; p=np.clip(d.p_ft.values,1e-6,1-1e-6)
    return roc_auc_score(y,p), log_loss(y,p)
try:
    ba,bl=m("mini_base"); aa,al=m("mini_adapt")
    print("=== ADAPTATION A/B (objective-grouped val) ===")
    print(f"baseline BERT-mini : AUC={ba:.4f} logloss={bl:.5f}")
    print(f"adapted  BERT-mini : AUC={aa:.4f} logloss={al:.5f}")
    print(f"external-data warmup effect: {aa-ba:+.4f} AUC, {bl-al:+.5f} logloss")
    print("=> MathDial+transcript pretraining HELPS" if aa>ba+0.002 else "=> pretraining does NOT clearly help")
except Exception as e:
    print("compare err", e)
PY
echo "PRETRAIN_CHAIN_DONE"
