# Container: ModernBERT ensemble (pre-trained + bundled → inference-only)

The local machine is an 8 GB M1 (no CUDA), so we cannot fine-tune a real transformer
locally. The **original** design trained ModernBERT *inside* the competition container
(A100, 6-hour budget) at submission time — a workaround for having no local GPU. **That
design is now superseded** (see Current status): the in-container fine-tune silently
trained on NaN and fell back to classical on every run. The **new** design
**pre-fine-tunes ModernBERT on a rented GPU, bundles the finished weights, and the
container runs INFERENCE ONLY** — it loads the bundled weights, ensembles the transformer
with the classical model, and writes predictions, all offline. It is still the project's
only path to the top of the leaderboard (the discrimination gap to #1 needs a transformer
representation — see [REVIEW_GUIDE.md §6](REVIEW_GUIDE.md)).

## Current status (important)
**The transformer has never actually contributed — and we now know exactly why.** The
public LB is unchanged at **#18 / ~429, best 0.6087** (id-1579); **every LB point to date
is classical-only.**

- **Root cause of the id-1579 fallback is now KNOWN** — it was never a `transformers`
  version issue. ModernBERT with `attn_implementation="sdpa"` produces **NaN logits on
  PADDED batches** (both bf16 and fp32; unpadded / equal-length batches are fine, which is
  exactly why it hid in the smoke tests). ModernBERT is built for **flash-attention**
  (which unpads); its SDPA fallback is broken for padding. The container tried `sdpa`
  first → trained on NaN → the `try/except` caught the garbage and fell back to classical.
  This happened on **every run for the entire competition**, so the "transformer leg" has
  contributed **exactly zero**. The 0.6087 gain was the **classical + a calibration prior
  fix**.
- **Fix = flash-attention** (`attn_implementation="flash_attention_2"`). Verified on real
  padded batches: finite, decreasing loss.
- **Once fixed, the transformer WORKS and is strong.** On the *real* ModernBERT-base
  (objective-grouped holdout, 10k subset, 3 epochs), the focused objective-centered
  representation scores **AUROC 0.6737** (classical OOF baseline 0.6446). Objective-grouped
  CV over-estimates the LB by ~0.04, so this is ≈**0.63 LB-equivalent discrimination vs
  classical 0.604** — a large, real gain. The earlier "signal is near-noise-saturated /
  build-complete" verdict was an artifact of a transformer that never ran.
- **Not yet submitted.** Being rebuilt inference-only in Session 2: domain-adaptive
  pretraining (DAPT) → local ship-gate → 5-fold OOF over the full 35k → if the gate passes,
  train a 5–6-seed ensemble (3 control + 3 history), bundle the weights, build the
  inference-only container, offline-smoke, submit.

## Design (inference-only + fallback-safe)
The bundle now ships **fine-tuned transformer weights**, not base weights to fine-tune, and
`main.py` does no training.
```
submission_container.zip (~large; committed as ~45 MB split parts)
├── main.py (= main_container.py)  # orchestrator (INFERENCE ONLY — no fine-tune)
├── features.py, model.py          # classical model (+ affine calibration) — the safety floor
├── dl_common.py                   # transcript->text builder (identical at train & inference)
├── dl_infer                       # load bundled weights + predict (replaces in-container dl_train.py)
└── assets/
    ├── artifacts.pkl              # pre-fit classical model
    ├── classical_oof.parquet      # classical objective-grouped OOF (ensemble weight fixed OFFLINE)
    ├── mbert_seeds/               # bundled fine-tuned ModernBERT weights (control + history seeds)
    └── flash_attn wheel           # pinned to the container's CUDA/torch
```

`main.py` flow (fully offline):
1. **Classical** predictions (always — the safety floor). The calibration is *neutralized*
   here so the classical leg is RAW (see calibration note below).
2. **Transformer:** **load the bundled fine-tuned weights and predict** (no training).
3. **Ensemble weight + calibration are decided OFFLINE**, not re-derived in-container: the
   local ship-gate (`solution/blend_gate.py`) fixes the blend weight and Platt/affine
   calibration from full-35k OOF (`classical_oof` + the transformer OOF). The container just
   applies them.
4. **Calibrate the final output once, post-blend:** `p → 0.685 + 0.68·(p − 0.7136)`
   (affine, so calibrating post-blend == per-leg). The pivot is the model's own OOF mean
   0.7136, not the train rate — the model over-predicts under objective shift.

**Container guards (the NaN is a *padding* bug, so inference also NaNs under sdpa):**
- Bundle a **flash-attn wheel pinned to the container's CUDA/torch** — the primary path.
- Coded **batch-size-1 sdpa fallback** (batch 1 = no padding = no NaN) if the wheel won't
  load.
- Keep the **classical leg + classical-only last-resort fallback**. Any exception → the
  submission can never score worse than the classical model.
- **Offline smoke-test the built zip** (`HF_HUB_OFFLINE=1`) before upload.

## The fixes from the adversarial review + the flash-attention fix
The headline fix is **flash-attention** — the sdpa-NaN correction that makes the transformer
leg *exist at all* (see Current status). On top of that, these principles from the
adversarial review and the Session-1 A/B carry into the offline GPU training / OOF:
1. **Focused objective-centered representation + left-truncation** (`dl_common.py`;
   `truncation_side="left"`, `max_len 3072`) — keep the session *ending* and the
   objective-relevant turns. This is a **hand-built attention prior**: the full-context
   (8192) representation was **decisively rejected** (−0.11 AUROC, 3.6× slower) because
   ModernBERT mean-pools and drowns the signal over ~5k irrelevant tokens. "Selection over
   coverage" is now a settled design principle.
2. **Platt-calibrate the DL leg** before blending (locally, this doubled the ensemble gain).
3. **AUROC-shaped ship-gate** — a discriminative-but-miscalibrated DL leg isn't rejected by
   a log-loss-only gate. Gate (OOF, full 35k, objective-grouped): blended AUROC ≥ classical
   + 0.015 **AND** calibrated logloss ≤ classical − 0.002.
4. **Doubly-disjoint val split** — hold out 15% of *objectives*, then move any session that
   touches a held-out objective wholly into val (multi-objective sessions share an identical
   transcript, so the objective-only split leaked siblings).
5. **Calibration prior 0.7025 → 0.7136** (the model's OOF mean).

## Local validation
- **Fallback path** verified from the actual built zip: forcing the DL to fail, the output
  matches the classical `submission_classical.zip` calibration exactly. So the floor is safe.
- **The silent-failure class is removed by design.** Because the weights are now
  pre-trained and bundled, the **EXACT shipped weights are validated on local OOF before
  upload** — the thing that hid the sdpa-NaN (in-container training we never observed) no
  longer exists. Nothing ships unless it clears the ship-gate above.
- **Remaining risk has shifted** from "ModernBERT accuracy is A100-only and unmeasured" to
  "container flash-attn / CUDA compatibility" — mitigated by the pinned wheel, the
  batch-size-1 sdpa fallback, the classical last-resort, and the offline smoke.

## The >50 MB upload (`automation/upload_big.py`)
Playwright-over-CDP caps uploads at 50 MB, so the large bundle uses
`launch_persistent_context` (Playwright-owned → no cap). The uploader was hardened after
several failures: it (a) **cancels any stuck in-progress job** first (a stuck job hides the
"New submission" button), (b) **retries opening the form**, and (c) **monitors from a
SEPARATE tab** — navigating the upload tab aborts the in-flight upload (the original bug that
made big uploads hang forever on "Uploading"). It then monitors to completion and reports
the score. Requires a manual re-login (the session cookie is memory-only).

## Rebuild + submit
```bash
# 1. OFFLINE (rented GPU): DAPT -> ship-gate -> 5-fold OOF -> 6-seed ensemble on all 35k
#    (solution/session2a.sh, solution/gpu_dapt.py, solution/gpu_oof.py, solution/blend_gate.py)
#    -> bundle the fine-tuned weights + transformer OOF into submission/assets/
# 2. Package (assets now hold mbert_seeds/ instead of train_texts.parquet):
python automation/package.py container            # -> submission_container.zip
HF_HUB_OFFLINE=1 python automation/test_submission_local.py   # offline smoke of the built zip
python automation/upload_big.py submission_container.zip smoke "smoke: validate inference path"
python automation/upload_big.py submission_container.zip normal "scored run"
# after clone, rebuild the zip from committed parts: ./reassemble_container.sh
```
