#!/bin/bash
# Session 2b (Fable's $3 confirmation): re-DAPT (base was lost when the pod terminated), then a
# 2nd-seed CONTROL-only OOF (seed_base=142). Averaged with seed-1's control OOF -> the pre-committed
# gate. History is dropped (OOF said it earned nothing). Produces /workspace/oof_control_s2.parquet.
set -u
cd /workspace/trace/solution
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DAPTDIR=/workspace/models/ModernBERT-dapt

echo "===== SESSION 2b start ====="
echo "----- DAPT (rebuild adapted base) -----"
python gpu_dapt.py "$DAPTDIR" 3072 1 || { echo "DAPT_FAILED"; exit 1; }

echo "----- 2nd-seed CONTROL OOF (seed_base=142) on DAPT'd base -----"
MBERT_DIR="$DAPTDIR" python gpu_oof.py control 2 /workspace/oof_control_s2.parquet 142 || { echo "OOF_FAILED"; exit 1; }

echo "SESSION2B_DONE"
