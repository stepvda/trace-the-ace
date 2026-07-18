#!/bin/bash
# Session 2a orchestrator (runs on the RunPod GPU): DAPT -> DAPT-gate -> OOF(control,history).
# Produces /workspace/oof_transformer.parquet for the local blend_gate.py.
set -u
cd /workspace/trace/solution
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STOCKDIR=/workspace/models/ModernBERT-base
DAPTDIR=/workspace/models/ModernBERT-dapt

echo "===== SESSION 2a start ====="

echo "----- [1/3] DAPT -----"
python gpu_dapt.py "$DAPTDIR" 3072 1 || { echo "DAPT_FAILED"; exit 1; }

echo "----- [2/3] DAPT gate (control: stock vs dapt, same split) -----"
MBERT_DIR="$STOCKDIR" python gpu_mbert.py control 10000 3 42 > /workspace/gate_stock.log 2>&1
MBERT_DIR="$DAPTDIR"  python gpu_mbert.py control 10000 3 42 > /workspace/gate_dapt.log  2>&1
STOCK=$(grep -aoE "ARM control: AUROC=[0-9.]+" /workspace/gate_stock.log | grep -oE "[0-9.]+$" | tail -1)
DAPT=$(grep -aoE  "ARM control: AUROC=[0-9.]+" /workspace/gate_dapt.log  | grep -oE "[0-9.]+$" | tail -1)
echo "DAPT-GATE: stock=$STOCK  dapt=$DAPT  (adopt dapt if dapt >= stock+0.004)"
BASE="$STOCKDIR"
if [ -n "${STOCK:-}" ] && [ -n "${DAPT:-}" ] && awk "BEGIN{exit !($DAPT >= $STOCK + 0.004)}"; then
  BASE="$DAPTDIR"; echo "DAPT ADOPTED -> base=$BASE"
else
  echo "DAPT not adopted -> base=$BASE (stock)"
fi

echo "----- [3/3] OOF (control,history) on $BASE -----"
MBERT_DIR="$BASE" python gpu_oof.py control,history 2 /workspace/oof_transformer.parquet || { echo "OOF_FAILED"; exit 1; }

echo "SESSION2A_DONE base=$BASE"
