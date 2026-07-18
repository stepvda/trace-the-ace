# Trace the Ace — Detailed Solution Documentation

This document describes the full solution end to end: the competition, the data,
every engineered feature, the models, the validation strategy, the runtime
compliance of the submission, and the browser automation used to download the
data and submit. It is intended to be self-contained and reproducible.

> **Status / two corrections to keep in mind while reading (this doc is the deep
> methodology reference; the fast overview is [REVIEW_GUIDE.md](REVIEW_GUIDE.md)):**
> - **Current best: public log loss 0.6087, rank #18** (this doc's feature/model
>   descriptions are current; the numbers in older prose may lag).
> - **Learning-objective target-encoding (§3.3) is DISABLED in the shipped model** — it is
>   leakage (huge in random CV, zero on *unseen* objectives, and it scored *worse* on the
>   leaderboard). The code remains as a `use_lo_target_enc` option (default `False`).
> - **A calibration step was added** after this doc was first written: the blended
>   probability is affinely recentered onto the estimated test base rate,
>   `p_final = 0.685 + 0.68·(p_raw − 0.7025)`, in `model.py:predict_pipeline` — this is what
>   took the score from 0.6144 to 0.6091. See [RESULTS_AND_STRATEGY.md](RESULTS_AND_STRATEGY.md).
> - **Validation groups by `learning_objective_id`** (the test holds out objectives), not by
>   session; and CV over-estimates the leaderboard because of the train→test shift.

---

## 1. Competition specification

| Item | Value |
|---|---|
| Competition | **Trace the Ace** (K-12 AI Infrastructure Program, operated by DrivenData) |
| URL | https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/ |
| Task | Binary classification: given a student–tutor lesson transcript + a short learning-objective description, predict whether the student answers the **next** assessment question on that objective correctly. |
| Target | `is_correct` ∈ {0.0, 1.0} (labels file column). Submission column is `probability` ∈ [0,1]. |
| Primary metric | **Log loss** (binary cross-entropy). Lower is better. |
| Secondary metric | ROC AUC (displayed only; does not affect ranking). |
| Data provenance | Real student–tutor conversations from Third Space Learning (TSL) and Eedi. |
| Submission type | **Code execution.** Upload `submission.zip` (root `main.py` + `assets/`); it runs offline in a container and writes `submission.csv`. |
| Runtime | Python 3.12; 24 vCPU / 220 GB RAM / 1×A100 80 GB; **no internet**; 60 GB zip limit; 6 h full / 10 min smoke; ≤500 log lines. Built with uv + PyTorch + vLLM (CUDA 12.9). |
| Submission limits | **3 full submissions / 7 days.** Smoke tests, cancelled, and failed jobs are **free**. |
| Deadline | Model submissions 2026-08-27 23:59 UTC; write-ups 2026-09-15. |

### Rules that shaped the design
- **No internet at inference** → all model weights are bundled in `assets/`; the
  stack is pure `scikit-learn`/`numpy`/`pandas`/`scipy` (guaranteed present, no
  downloads).
- **Test samples processed independently; no cross-sample learning** → every
  transformer (TF-IDF, SVD, scaler, target-encoding map, models) is **fit on
  training data only** and merely *applied* at inference. Nothing is fit on test.
- **Do not print/log any test-data information** (counts, means, excerpts) →
  `main.py` logs only generic progress strings.

---

## 2. Data

### 2.1 Files (downloaded from the authenticated *Data download* page)
```
train_features.csv            35,072 rows
train_labels.csv              35,072 rows
train_transcripts.zip     ->  22,821 per-session CSVs (~576 MB unzipped)
submission_format_full.csv    10,508 rows  (full public test)
submission_format_smoke.csv      100 rows  (smoke = subset of training)
```

### 2.2 Schemas (as actually delivered — note deltas from the docs)
`train_features.csv`
| column | type | notes |
|---|---|---|
| `response_id` | str | unique sample id |
| `session_id` | str | links to `train_transcripts/{session_id}.csv` |
| `learning_objective_id` | str | **present in data, not in the docs**; 1:1 with the text below |
| `learning_objective` | str | short description of the assessed objective |

`train_labels.csv`
| column | type | notes |
|---|---|---|
| `response_id` | str | |
| `is_correct` | float | **column is `is_correct`, not `correct` as the docs say** |

`train_transcripts/{session_id}.csv` (one row per utterance)
| column | type | notes |
|---|---|---|
| `session_id` | str | |
| `utterance_id` | str | order within session |
| `role` | str | `tutor` / `student` / **`background`** (e.g. `[unclear]`) |
| `content` | str | transcribed spoken dialogue |
| `timestamp` | str | **elapsed `HH:MM:SS`**, not an absolute datetime |

### 2.3 EDA findings
- **Label base rate 0.7025** (24,637 correct / 10,435 incorrect). Constant-mean
  prediction gives log loss **0.60876** on the *train* rate. NOTE: this is a train-set
  quantity and was **never observed on the LB** — the true test-constant is ≈**0.6236**
  (test base rate ≈0.685). Keep 0.60876 only as a CV reference, not "the bar to beat."
- **398** distinct learning objectives (id ↔ text is 1:1). Counts are very
  skewed: median 9 samples/objective, max 1,373; 159 objectives have ≥20.
- Sessions are **long**: ~264 utterances each on average (max 469 in sample).
- Responses per session: mean 1.54, max 10; 8,364 sessions have >1 response
  (same transcript, different objective).
- Individual feature↔label correlations are **weak** (|r| ≤ ~0.09): the strongest
  raw signals are tutor praise, amount of student talk, and turn-switching. The
  task is genuinely hard, so text + objective difficulty + model interactions do
  the heavy lifting.
- **Full-test response_ids do not appear in train** (true holdout); smoke-test
  response_ids are 100% from train (as documented). Test sessions are unseen →
  validation must estimate generalization to **new sessions**.

---

## 3. Feature engineering (`solution/features.py`)

`build_features(features_df, transcripts_dir)` returns a table indexed by
`response_id`. Session-level features are computed **once per session** (cached)
because responses in the same session share one transcript, then joined to each
response. All numeric NaNs are handled downstream by median imputation (LR) or
natively (HistGradientBoosting). The module is imported by **both** training and
inference so the computation is identical.

### 3.1 Numeric features (40)
Grouped by intuition; all are per-session unless noted.

**Volume / participation**
| feature | meaning |
|---|---|
| `n_utt` | total utterances |
| `n_student`, `n_tutor` | utterances by each role |
| `frac_student_utt` | share of utterances spoken by the student |
| `tot_words`, `stud_words`, `tut_words` | word counts (total / student / tutor) |
| `frac_words_student` | student share of all words |
| `words_ratio_st` | student words ÷ tutor words |
| `mean_words_utt` | mean words per utterance |

**Utterance length distribution**
| feature | meaning |
|---|---|
| `mean_words_student`, `mean_words_tutor` | mean words per utterance by role |
| `max_words_student`, `max_words_tutor` | longest utterance by role |
| `std_words_student`, `median_words_student` | spread of student utterance lengths |

**Timing (from `HH:MM:SS`)**
| feature | meaning |
|---|---|
| `duration_sec` | session length (last − first timestamp) |
| `mean_gap_sec`, `median_gap_sec`, `max_gap_sec` | inter-utterance gaps |
| `words_per_min` | pace = total words ÷ minutes |

**Interaction dynamics**
| feature | meaning |
|---|---|
| `turn_switches` | number of role changes between consecutive utterances |
| `switch_rate` | switches ÷ utterances (dialogue back-and-forth density) |

**Pedagogical language (lexicon counts + per-utterance rates)**
| feature | meaning |
|---|---|
| `tutor_praise`, `tutor_praise_rate` | tutor positive-feedback phrases ("well done", "exactly", "correct", …) |
| `tutor_corrective`, `tutor_corrective_rate` | tutor corrective phrases ("not quite", "try again", "almost", …) |
| `student_uncertain`, `student_uncertain_rate` | student uncertainty markers ("i don't know", "not sure", "confused", …) |
| `student_affirm`, `student_affirm_rate` | student affirmations ("got it", "makes sense", "i understand", …) |
| `tutor_q`, `student_q`, `tutor_q_rate`, `student_q_rate` | question-mark counts and rates by role |

**Student cognitive signal**
| feature | meaning |
|---|---|
| `student_nums` | digit count in student speech (numeric reasoning, math relevance) |
| `student_uniq_ratio` | unique/total student tokens (vocabulary richness vs. repetition) |
| `first_student_words`, `last_student_words` | length of the student's first / last utterance |
| `last_role_student` | 1 if the session ends on a student turn |

The lexicons (`TUTOR_PRAISE`, `TUTOR_CORRECTIVE`, `STUDENT_UNCERTAIN`,
`STUDENT_AFFIRM`) are defined at the top of `features.py`; matching is
lowercase substring counting over the concatenated role text.

### 3.2 Text features
- `text_all` — full transcript content → **TF-IDF** (word 1–2 grams, sublinear
  tf, `min_df=5`, `max_features=30000`).
- `text_lo` — learning-objective description → TF-IDF (1–2 grams, `max_features=3000`).
- (`text_student` / `text_tutor` are produced and available; the default config
  disables the separate student vectorizer since `text_all` already contains
  student utterances — see `use_student_tfidf`.)

### 3.3 Learning-objective target encoding
Some objectives are intrinsically harder. We encode each `learning_objective_id`
by its **smoothed** historical correctness:
`enc(o) = (Σ_train y_o + m·μ) / (n_o + m)` with smoothing `m=20` and global mean
`μ`. Unseen objectives fall back to `μ`. This is computed **only on the training
fold** during CV (no leakage) and on all training data for the final model; at
inference it is a pure lookup, so each test sample is scored independently.

---

## 4. Models (`solution/model.py`)

Two complementary learners are blended:

1. **Logistic Regression** (`liblinear`, `C=1.0`) on the sparse design matrix
   `[TF-IDF(text_all) | TF-IDF(text_lo) | standardized numeric | objective encoding]`.
   Strong, well-calibrated linear model for high-dimensional text → log loss.
2. **HistGradientBoostingClassifier** (400 iters, lr 0.06, 31 leaves, L2=1,
   `min_samples_leaf=40`, early stopping) on dense
   `[raw numeric (NaN-aware) | SVD_120(TF-IDF) | objective encoding]`.
   Captures non-linear interactions among the structural/pedagogical features.

**Blend**: `p = w·p_LR + (1−w)·p_HGB`, with `w` chosen on out-of-fold predictions
(default 0.55). **Clipping** to `[c, 1−c]` (default `c=0.005`) caps overconfident
predictions, which log loss punishes heavily; the level is chosen on OOF.

`fit_pipeline(config, X, y)` fits every component and returns a single picklable
artifacts dict; `predict_pipeline(artifacts, X)` applies them. Both CV and
inference call the *same* `predict_pipeline`, guaranteeing train/inference parity.

---

## 5. Validation (`solution/cv_fast.py`)

- **StratifiedGroupKFold (5 folds), grouped by `session_id`**, stratified on the
  label. Grouping by session prevents a session's transcript from appearing in
  both train and validation — matching the real test, whose sessions are unseen.
- **Speed design**: TF-IDF, SVD and the numeric scaler are **unsupervised**, so
  they are fit **once on all training data** and reused across folds; only the
  **supervised** parts (LR, HGB, and the objective target-encoding) are refit per
  fold. This changes CV wall-clock from ~25 min to a few minutes and introduces
  only negligible (label-free) leakage into the CV estimate; the **final model is
  strictly clean** because `fit_pipeline` fits everything on training data that
  the held-out test never sees.
- The CV run also searches the best blend weight and clip level on OOF predictions.

### 5.1 Results
5-fold StratifiedGroupKFold (grouped by `session_id`), out-of-fold log loss:

| Model | OOF log loss | Δ vs baseline |
|---|---|---|
| Constant mean (base rate 0.70) | 0.60876 | — |
| HistGradientBoosting | 0.54349 | −0.0653 |
| Logistic Regression (TF-IDF + numeric + objective enc) | 0.53541 | −0.0734 |
| **Blend, weight `w_LR = 0.70`** | **0.53386** | **−0.0749 (≈12%)** |

- OOF **AUC = 0.735**.
- The LR is the stronger single model; the HGB adds a small amount of
  decorrelated signal, so the OOF-optimal blend leans 70% LR / 30% HGB.
- **Probability clipping gave no improvement** (the two-model blend rarely emits
  values outside ~[0.02, 0.98]), so the final clip is effectively inert and kept
  only as a numerical safety bound.
- Per-feature label correlations are weak, so ~0.075 of log loss comes from the
  *combination* of transcript text, structural/pedagogical features, and
  smoothed objective difficulty — no single feature dominates.

---

## 6. Final model & artifacts (`solution/train_final.py`)

Fits `fit_pipeline` on **all 35,072** training rows and saves
`submission/assets/artifacts.pkl` (fitted vectorizers, SVD, scaler + medians,
objective-encoding map + global mean, LR, HGB, and the config incl. blend weight
and clip). Saved with `joblib` (compressed).

---

## 7. Inference (`submission/main.py`) & runtime compliance

Flow: read `data/test_features.csv`, `data/submission_format.csv`,
`data/test_transcripts/` → `build_features` (identical to training) →
`predict_pipeline` → align to the submission format's `response_id` order →
clip to `[1e-4, 1−1e-4]` → write `submission.csv` next to `main.py`.

Compliance:
- **Offline**: no network calls; all weights in `assets/`; pure sklearn stack.
- **Independent test samples**: no fitting on test; objective encoding is a lookup.
- **No test-data logging**: only generic messages ("Building features…",
  "Running inference…", "Wrote submission.csv").
- **Robustness**: missing transcript files → zero-count features; any missing
  prediction falls back to the training base rate; probabilities always in (0,1).

---

## 8. Submission packaging (`automation/package.py`)

Builds `submission.zip` with `main.py`, `features.py`, `model.py` at the **root**
(verified in-code that `main.py` is at the archive root, per the rules) and the
`assets/` folder. Local end-to-end validation (`automation/test_submission_local.py`)
recreates the exact container layout from a slice of training data, runs
`main.py` as a subprocess, and asserts the output schema/row-set/value ranges
before anything is uploaded.

---

## 9. Browser automation (`automation/`)

The submission and data download require the authenticated site. Because
Edge 150 disables remote-debugging on the **default** profile (Chrome ≥136
security change), a fresh **visible** Edge is launched on a dedicated profile
with `--remote-debugging-port=9333`; the user logs in once; Playwright then
attaches over **CDP** (`connect_over_cdp`) — never headless, and driving a real
browser window. Scripts:
- `crawl.py` — capture all competition pages (problem, rules, formats, etc.).
- `get_data_urls.py` — read fresh signed S3 links from *Data download*.
- `submit_code.py` — open the *Code jobs* form, attach the zip, pick
  **Smoke test** or **Normal submission**, submit.
- `poll_codejobs.py` — read job + scored-submission status.

**Submission protocol**: run a **free smoke test** first (validates that the zip
runs in the real container and that the sklearn stack is available), then, only
after it passes, spend one of the 3 weekly **Normal** submissions.

---

## 10. Reproduce

```bash
python solution/build_cache.py               # cache engineered features (~90s)
python solution/cv_fast.py                   # cross-validated log loss
python solution/train_final.py               # fit final -> assets/artifacts.pkl
python automation/package.py                 # -> submission.zip
python automation/test_submission_local.py   # end-to-end local runtime check
# then, via the logged-in Edge:
python automation/submit_code.py submission.zip smoke     # free validation
python automation/submit_code.py submission.zip normal    # scored submission
```

## 11. Limitations & future work
- Multi-objective sessions share one transcript, so within-session responses get
  identical structural features; segmenting the transcript per objective adds
  signal. This idea is now realized in the transformer leg's **objective-centered
  representation** (see §12).
- The pedagogical lexicons are hand-built and English/TSL-specific; learned
  representations (e.g. bundled sentence embeddings, allowed offline) could
  generalize better.
- The blended LR + HGB stack above is the **only leg that has ever scored on the
  leaderboard**; every LB number in this doc (incl. 0.6087 / #18) is
  **classical-only**. A transformer leg is under active development (§12) but is
  **not yet submitted**.

## 12. Transformer leg (ModernBERT) — under development, not yet submitted

> Everything below is in-progress work on a **separate model leg**, tracked in
> [RESULTS_AND_STRATEGY.md](RESULTS_AND_STRATEGY.md). It has contributed **nothing**
> to the public LB to date; the classical stack (§§3–10) is what ships today.

The task's remaining gap is **discrimination**, not calibration (classical OOF
AUROC ≈ 0.604; #1 ≈ 0.63). A fine-tuned transformer over the transcript is the
lever for that gap.

- **The critical bug — the transformer never actually trained.** ModernBERT with
  `attn_implementation="sdpa"` emits **NaN logits on padded batches** (both bf16 and
  fp32; unpadded/equal-length batches are fine, so it passed smoke tests). ModernBERT
  is designed for **flash-attention** (which unpads); its SDPA fallback is broken for
  padding. The old container trainer (`solution/dl_train.py`) tried `sdpa` first, trained
  on NaN, and **silently fell back to the classical prediction every run** — so the
  "transformer leg" had contributed **zero** for the entire competition (the
  long-unexplained id-1579 fallback).
- **The fix** is `attn_implementation="flash_attention_2"` (see `solution/gpu_mbert.py`,
  `solution/gpu_dapt.py`, `solution/verify_flash.py`); padded batches now train with
  finite, decreasing loss. Once fixed, the model is strong: a 3-arm A/B on real
  ModernBERT-base (objective-grouped holdout, 10k subset, 3 epochs) gave the
  objective-centered "control" rep **AUROC 0.6737** vs the classical OOF baseline
  0.6446. Objective-grouped CV over-estimates the LB by ~0.04, so control ≈ **0.63
  LB-equivalent** — a large, real discrimination gain.
- **Representation: selection over coverage (settled).** The **objective-centered
  representation** — a focused window around the assessed objective's turns
  (`solution/dl_common.py`, `RELEVANT_WORDS`/`RECENT_WORDS`, optional additive
  `HISTORY_WORDS`) — decisively beats highlighting the **whole** 8192-token transcript
  (`full_context` mode: −0.11 AUROC, 3.6× slower, flat learning curve). ModernBERT
  mean-pools, so pooling over ~5k mostly-irrelevant tokens drowns the signal; the
  focused window is a hand-built attention prior the model can't cheaply relearn.
- **Architecture: pre-train-and-bundle → inference-only container.** In-container
  fine-tuning at submission time is **superseded**: it was a no-local-GPU workaround
  and the thing that hid the silent NaN failure. The new design fine-tunes on a
  **rented GPU** (RunPod RTX 4090), validates the **exact shipped weights** on local OOF,
  bundles them into `assets/`, and the container runs **inference only** — removing the
  6 h training ceiling and the silent-failure class. Container guards: a pinned
  flash-attn wheel plus a coded **batch-size-1 sdpa fallback** (batch 1 = no padding = no
  NaN), and the classical leg kept as a last-resort fallback; the built zip is
  offline-smoke-tested (`HF_HUB_OFFLINE=1`) before upload.
- **Note on DAPT:** domain-adaptive pretraining was only ever applied to a local
  **DistilBERT proxy** (`cache/distilbert_adapted`), never the encoder that ships (stock
  **ModernBERT-base**). DAPT on the transcript corpus (`solution/gpu_dapt.py`) is now
  being evaluated over the full-35k OOF (`solution/gpu_oof.py`, `solution/blend_gate.py`).
- **Ship gate (OOF, full 35k, objective-grouped):** the blended (transformer +
  classical) OOF must beat classical AUROC by **≥ 0.015** *and* the calibrated log loss
  by **≥ 0.002** before a transformer submission is spent.
- **Related dead-end vs still-open idea:** a zero-shot LLM *extractor* (verdicts from a
  frozen model) was a dead end and stays retired; a QLoRA decoder *classifier*
  fine-tuned on the labels over the focused representation is a **different, still-open**
  moonshot idea, not yet attempted.
