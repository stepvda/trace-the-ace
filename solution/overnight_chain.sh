#!/bin/bash
# Autonomous overnight ensemble build (RAM-safe: stages run sequentially).
cd /Users/nstephane/Dev/AI_Data_Science_training/trace-the-ace
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false

echo "=== STAGE 1: wait for ELECTRA v1 ==="
until [ -f solution/cache/ft_val_electra_v1.csv ] || grep -qE "Traceback|RuntimeError" /tmp/ft_v1.log 2>/dev/null; do sleep 20; done
echo "v1 result:"; grep -aE "\"eval_auc\"|AUC=" /tmp/ft_v1.log | tail -2

echo "=== STAGE 2: ensemble eval (classical + v1) ==="
$PY solution/ensemble_eval.py electra_v1 2>&1 | grep -vE "Loading|it/s" || true

echo "=== STAGE 3: classical diversity OOF ==="
$PY solution/gen_diverse_oof.py 2>&1 | tail -6 || true

echo "=== STAGE 4: ELECTRA v2 (longer context, diversity) ==="
$PY -u solution/finetune.py --model google/electra-small-discriminator \
   --n_words 600 --max_len 512 --batch 6 --epochs 3 --out_tag electra_v2 2>&1 \
   | grep -aE "train=|\"eval_auc\"|AUC=|Error|RuntimeError" | tail -4 || true

echo "=== STAGE 5: final combined robust ensemble ==="
$PY solution/final_ensemble.py 2>&1 | tail -20 || true
echo "OVERNIGHT_CHAIN_DONE"
