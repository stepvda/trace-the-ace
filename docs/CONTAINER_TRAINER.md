# A100 Container-Trainer (ModernBERT ensemble)

The local machine is an 8 GB M1 (no CUDA), so the only way to get GPU training is to
**train inside the competition container** — which has an A100 (80 GB) and a 6-hour
budget. This submission bundles the training data + a fine-tune script; `main.py` trains a
transformer on the A100, ensembles it with the classical model, and writes predictions —
all offline. **It is the project's only path to the top of the leaderboard** (the
discrimination gap to #1 needs a transformer representation — see
[REVIEW_GUIDE.md §6](REVIEW_GUIDE.md)).

## Current status (important)
The container ran once as a scored submission (id-1579 → **0.6087**), but the **transformer
leg fell back to classical** — the run finished in ~11 minutes, far too fast for a real
fine-tune, so the DL step errored into the try/except. The 0.6087 gain came from the
**classical + a calibration prior fix**, *not* the transformer. So the ModernBERT accuracy
has **never actually been measured on any hardware**. The likely cause is a
`transformers`/ModernBERT runtime/version issue in the container (flagged in review;
`transformers>=4.48` is required for ModernBERT and was unverified in the real runtime).
**Next step: read the id-1579 container log, fix the runtime issue, re-run.**

## Design (self-optimizing + fallback-safe)
```
submission_container.zip (~700 MB; committed as 45 MB split parts)
├── main.py (= main_container.py)  # orchestrator
├── features.py, model.py          # classical model (+ affine calibration)
├── dl_common.py                   # transcript->text builder (identical train & inference)
├── dl_train.py                    # torch-only in-container fine-tune + predict + held-out eval
└── assets/
    ├── artifacts.pkl              # pre-fit classical model
    ├── train_texts.parquet        # 35k recency texts + labels + session_id (avoids bundling raw transcripts)
    ├── classical_oof.parquet      # classical objective-grouped OOF (for ensemble weighting)
    └── base_model/                # ModernBERT-base weights, offline
```

`main.py` flow (fully offline):
1. **Classical** predictions (always — the safety floor). The calibration is *neutralized*
   here so the classical leg is RAW (see calibration note below).
2. **Transformer:** fine-tune ModernBERT on `train_texts` on the GPU, holding out 15% of
   **sessions** (session-grouped — avoids the sibling leak, see fixes).
3. **Platt-calibrate the DL leg** on a disjoint half of the val, then **choose the ensemble
   weight** on the other half.
4. **Ensemble weight gate:** accept a weight only if the blend beats classical by >0.002 log
   loss OR improves val AUROC by >0.005 (a transformer's value is AUROC-shaped; a log-loss-
   only gate can wrongly reject it). Weight capped ≤0.6. If neither, weight = 0 → pure
   classical. Any exception → try/except fallback to classical. **The submission can never
   score worse than the classical model.**
5. **Calibrate the final output once, post-blend:** `p → 0.685 + 0.68·(p − 0.7136)` (affine,
   so calibrating post-blend == per-leg; ensemble and fallback both get it). The pivot is the
   model's own OOF mean 0.7136, not the train rate — the model over-predicts under objective
   shift.

`dl_train.py` uses only **torch + transformers** (no accelerate/datasets). GPU-aware (CUDA
in the container; CPU/MPS locally), respects a wall-clock budget, and has a **smoke mode**
so the free smoke test validates the code path.

## The 5 fixes from the adversarial review (why this build differs from the first)
An adversarial code review found the original container would likely null the transformer.
This build fixes:
1. **Left-truncation** (`dl_train.py`, `tok.truncation_side="left"`, `max_len=3072`) — keep
   the session *ending* (where the outcome-relevant content is); the default right-truncation
   cut it off.
2. **Platt-calibrate the DL leg** before blending (locally, this doubled the ensemble gain).
3. **AUROC-shaped gate** (step 4) — so a discriminative-but-miscalibrated DL leg isn't
   rejected by a log-loss-only gate, wasting the A100 slot.
4. **Session-grouped val split** — the objective-only split leaked siblings (multi-objective
   sessions have identical text).
5. **Calibration prior 0.7025 → 0.7136** (the model's OOF mean).

## Local validation
- **Fallback path** verified from the actual built zip: forcing the DL to fail, the output
  matches the classical `submission_classical.zip` calibration exactly. So the floor is safe.
- The **Platt + AUROC-gate logic** is unit-tested on synthetic data.
- The **ModernBERT DL path is A100-only** and unvalidated (the M1 swaps). This is the open risk.

## The >50 MB upload (`automation/upload_big.py`)
Playwright-over-CDP caps uploads at 50 MB, so the ~700 MB bundle uses
`launch_persistent_context` (Playwright-owned → no cap). The uploader was hardened after
several failures: it (a) **cancels any stuck in-progress job** first (a stuck job hides the
"New submission" button), (b) **retries opening the form**, and (c) **monitors from a
SEPARATE tab** — navigating the upload tab aborts the in-flight upload (the original bug that
made big uploads hang forever on "Uploading"). It then monitors to completion and reports
the score. Requires a manual re-login (the session cookie is memory-only).

## Rebuild + submit
```bash
# train_texts.parquet + classical_oof.parquet are prebuilt in submission/assets/
python automation/package.py container            # -> submission_container.zip
python automation/upload_big.py submission_container.zip smoke "smoke: validate DL path"
python automation/upload_big.py submission_container.zip normal "scored run"
# after clone, rebuild the zip from committed parts: ./reassemble_container.sh
```
