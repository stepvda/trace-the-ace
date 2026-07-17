#!/bin/bash
# Autonomous PoC chain: wait for cache -> embed -> compare feature sets (objective-grouped).
set -e
cd /Users/nstephane/Dev/AI_Data_Science_training/trace-the-ace
PY=.venv/bin/python
CACHE=solution/cache

echo "=== [1] waiting for prep cache ==="
until [ -f "$CACHE/tfidf.npz" ] && [ -f "$CACHE/svd256.npy" ] && [ -f "$CACHE/row_ids.csv" ]; do sleep 5; done
# ensure prep process finished writing (svd is last)
sleep 3
echo "cache ready: $(ls -1 $CACHE | tr '\n' ' ')"

echo "=== [2] extracting frozen embeddings (bge-small) ==="
$PY -u solution/embed_transcripts.py --model BAAI/bge-small-en-v1.5 --split train --n_words 350 --batch 64

TAG=bge-small-en-v1.5
echo "=== [3] objective-grouped experiments ==="
echo "--- classical (tfidf+svd+numeric) ---"
$PY solution/experiment.py --group objective --lr_C 1.0
echo "--- classical + embeddings ---"
$PY solution/experiment.py --group objective --lr_C 1.0 --use_emb --emb_tag $TAG
echo "--- embeddings-only (no tfidf) ---"
$PY solution/experiment.py --group objective --lr_C 1.0 --use_emb --emb_tag $TAG --no_tfidf
echo "=== [4] session-grouped sanity (classical + emb) ==="
$PY solution/experiment.py --group session --lr_C 1.0 --use_emb --emb_tag $TAG
echo "POC_DONE"
