# A100 Container-Trainer (ambitious July-14 submission)

Since the local machine is an 8 GB M1 (no CUDA, can't fine-tune real
transformers), the only way to get GPU training is to **train inside the
competition container**, which has an A100 (80 GB) and a 6-hour budget. This
submission bundles the training data and a fine-tuning script; `main.py` trains a
transformer on the A100, ensembles it with the classical model, and writes
predictions — all offline.

## Design (self-optimizing + safe)
```
submission.zip (692 MB)
├── main.py            # orchestrator
├── features.py        # classical feature engineering
├── model.py           # classical fit/predict
├── dl_common.py       # shared transcript->text builder (train == inference)
├── dl_train.py        # torch-only in-container fine-tune + predict + held-out eval
└── assets/
    ├── artifacts.pkl        # pre-fit classical model (49 MB)
    ├── train_texts.parquet  # 35k recency texts + labels (134 MB, avoids bundling 576MB transcripts)
    ├── classical_oof.parquet# classical objective-grouped OOF (for ensemble weighting)
    └── base_model/          # ModernBERT-base weights, offline (574 MB, 8192-token context)
```

`main.py` flow (fully offline):
1. **Classical** predictions on the test set (always; the safety floor).
2. **Transformer**: fine-tune ModernBERT on `train_texts` on the GPU, holding out
   15% of **learning objectives** (leakage-free val).
3. **Ensemble weight chosen inside the container** by comparing DL vs. classical
   on that held-out split (classical val = bundled OOF). If the DL doesn't
   generalize, the weight is 0 → pure classical. If the DL step throws, a
   try/except falls back to classical. **The submission can never score worse
   than the classical model** because of this measured, self-optimizing weight.
4. Write `submission.csv` (full confidence — leaderboard evidence showed
   shrink-to-prior *hurts* this model).

`dl_train.py` uses only **torch + transformers** (no accelerate/datasets) to
minimize container package requirements. It is GPU-aware (CUDA in the container,
CPU/MPS fallback locally), respects a wall-clock budget, and has a **smoke mode**
(tiny subset) so the free smoke test validates the whole path in ~10 minutes.

## Validation status
- **Locally validated** (fast model swap via `BASE_MODEL_DIR`): `dl_train` loop
  trains + predicts; `main.py` runs end-to-end, computes the ensemble weight on
  the held-out split, and falls back to classical correctly.
- **ModernBERT on the A100**: validated via the free **smoke test** in the real
  container (the M1 can't run ModernBERT — it swaps). Only the *full-training
  accuracy* is unknown until a real (non-smoke) submission on July 14.

## The >50 MB upload workaround
Playwright connected **over CDP** caps file uploads at 50 MB ("browser not
co-located"). The 692 MB bundle therefore cannot be uploaded through the
CDP-attached Edge. Fix: **`automation/upload_big.py`** launches the *same logged-in
automation profile* via `launch_persistent_context` (Playwright-owned → no cap),
uploads, and monitors. (This briefly closes/reopens the automation window; the
login persists because it reuses the same profile directory.)

## July-14 plan (submissions reset; 3 available)
The **best-understood** submission is still the **classical unshrunk** model
(`submission.zip` in the `main`-line build; ~0.607 est.). Spend the 3 submissions:
1. **Classical, `shrink_a=1.0`** (reliable ~0.607 — the safe improvement over the
   current 0.6144).
2. **Container-trainer** (this bundle) as a **full** (non-smoke) run — the DL
   ensembles in only if it beats classical on the held-out split, so downside is
   the classical floor; upside is a real transformer on the A100.
3. **Classical, `shrink_a=0.7`** (or bracket the calibration optimum), based on
   what #1/#2 reveal.

To regenerate the container-trainer bundle:
```bash
python solution/build_dl_texts.py         # train_texts.parquet
python solution/gen_classical_oof.py       # classical_oof.parquet
# ModernBERT is already saved in submission/assets/base_model/
python automation/package.py               # -> submission.zip (692 MB)
python automation/upload_big.py submission.zip normal "container-trainer full run"
```
