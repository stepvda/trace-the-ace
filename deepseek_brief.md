# TRACE THE ACE — full project brief for DeepSeek consultation



===== FILE: README.md =====
# Trace the Ace — Tutoring Outcomes Prediction

Solution for the DrivenData / K-12 AI Infrastructure competition
[*Trace the Ace*](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/).

**Task.** Given a student–tutor lesson transcript and a short learning-objective
description, predict the probability that the student answers the *next*
assessment question on that objective correctly (`is_correct` ∈ {0,1}).

**Metric.** Log loss (binary cross-entropy); AUC shown for reference only.
Constant-mean baseline log loss = **0.6088** (train base rate 0.70).

**Submission type.** Code execution: a `submission.zip` with `main.py` at the
root + `assets/`, run offline in a container (Python 3.12, no internet), reading
`data/test_features.csv` + `data/test_transcripts/` and writing `submission.csv`.

## Approach

Pure `scikit-learn` (guaranteed available in the offline runtime; no external
downloads) blending two complementary models, validated with **StratifiedGroupKFold
grouped by `session_id`** (the hidden test set has unseen sessions):

1. **Feature engineering** (`solution/features.py`, shared by train & inference):
   - Transcript structure: utterance/word counts, tutor/student balance, turn
     switching, timing (duration, gaps) parsed from the `HH:MM:SS` timestamps.
   - Pedagogical signals: tutor praise vs. corrective language, student
     uncertainty vs. affirmation markers, question rates, lexical richness,
     first/last student-utterance sizes.
   - Text: TF-IDF over the full transcript and over the learning-objective text.
   - Learning-objective difficulty via **smoothed, CV-safe target encoding**.
2. **Models**: Logistic Regression on the sparse TF-IDF + numeric features, and
   a HistGradientBoosting classifier on numeric + SVD(TF-IDF) + objective
   encoding. Blended and lightly clipped to optimize log loss.

## Layout

```
data/                     downloaded competition data (gitignored)
solution/
  features.py             feature engineering (train + inference)
  model.py                fit_pipeline / predict_pipeline + config
  build_cache.py          build & cache train features
  cv_fast.py              fast StratifiedGroupKFold CV
  train_final.py          fit on all data -> submission/assets/artifacts.pkl
submission/
  main.py                 offline inference entrypoint
  assets/artifacts.pkl    pickled fitted pipeline
automation/               Playwright/CDP scripts driving the logged-in Edge
  package.py              build submission.zip
  submit_code.py          upload a code job (smoke/normal)
  poll_codejobs.py        read job + submission status
```

## Result

**Rank #27 / 329** (up from #79), best public Log Loss **0.6144**. The whole field
sits within ~0.008 of the constant baseline (0.6088); leaderboard #1 = 0.6013.
Full journey, learnings, and the calibration lesson: **[docs/RESULTS_AND_STRATEGY.md](docs/RESULTS_AND_STRATEGY.md)**.

Two submissions are prepared for the next window (resets **2026-07-14**, 3/week):
- `submission_classical.zip` (49 MB) — classical, **unshrunk** (est. ~0.607, the
  reliable improvement). Entrypoint `submission/main.py`.
- `submission_container.zip` (692 MB) — **A100 container-trainer**: fine-tunes
  ModernBERT in-container and ensembles it with the classical model, weighted by a
  held-out split, with a classical fallback. Entrypoint `submission/main_container.py`.
  See **[docs/CONTAINER_TRAINER.md](docs/CONTAINER_TRAINER.md)**.

## Reproduce

```bash
python solution/build_cache.py       # cache engineered features (~90s)
python solution/cv_fast.py           # cross-validated log loss (session-grouped)
GROUP_BY=objective python solution/cv_fast.py   # leakage-free (objective-grouped) CV
python solution/train_final.py <cfg.json>       # fit final -> assets/artifacts.pkl
python automation/package.py classical           # -> submission_classical.zip
python automation/test_submission_local.py 800   # end-to-end local runtime check
# container-trainer build:
python solution/build_dl_texts.py && python solution/gen_classical_oof.py
python automation/package.py container            # -> submission_container.zip
```

## Docs
- [docs/SOLUTION.md](docs/SOLUTION.md) — full methodology, data schema, every feature.
- [docs/RESULTS_AND_STRATEGY.md](docs/RESULTS_AND_STRATEGY.md) — leaderboard journey, learnings, July-14 plan.
- [docs/CONTAINER_TRAINER.md](docs/CONTAINER_TRAINER.md) — the A100 transformer-ensemble submission.



===== FILE: docs/SOLUTION.md =====
# Trace the Ace — Detailed Solution Documentation

This document describes the full solution end to end: the competition, the data,
every engineered feature, the models, the validation strategy, the runtime
compliance of the submission, and the browser automation used to download the
data and submit. It is intended to be self-contained and reproducible.

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
  prediction gives log loss **0.60876** — the number any real model must beat.
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
  identical structural features; segmenting the transcript per objective could add
  signal.
- The pedagogical lexicons are hand-built and English/TSL-specific; learned
  representations (e.g. bundled sentence embeddings, allowed offline) could
  generalize better.
- Only shallow models are used for portability; the A100 permits bundling a small
  transformer for the transcripts if higher accuracy is needed.



===== FILE: docs/EXPERIMENT_LOG.md =====
# Experiment Log — what helped, what hurt

A running, evidence-based record of every measure tried on *Trace the Ace*, with
its **measured effect**. Metrics are **objective-grouped CV** (leakage-free,
hold out learning objectives) unless a **leaderboard (LB)** score is given.
Baselines: constant-mean LB ≈ **0.6088**; field #1 = 0.6013; field AUROC ≈ 0.63.

> **Read this first — the core lesson.** Local CV *systematically over-estimates*
> the leaderboard because the train→test distribution shift (TSL + Eedi providers)
> is not present in any split of the training data. So CV is trustworthy for
> **relative** comparisons (A vs B on held-out objectives) but **not** for
> predicting the absolute LB score. Session-grouped CV was *dangerously*
> optimistic (0.534) and produced my worst submission.

---

## Leaderboard submission history (the only ground truth)
| Job | Model | shrink_a | LB Log Loss | Rank | Verdict |
|---|---|---|---|---|---|
| id-1002 | TF-IDF + **LO target-encoding** (leaky) + numeric | 1.0 | 0.6224 | #79 | ❌ worse than baseline |
| **id-1019** | enriched feats, **no LO-enc**, keep LO-tfidf | 0.4 | **0.6144** | **#27** | ✅ best |
| id-1022 | same model, more shrinkage | 0.12 | 0.6200 | #27 | ❌ over-shrunk |

---

## ✅ What HELPED
| Measure | Effect | Evidence |
|---|---|---|
| **Objective-grouped CV** (vs session-grouped) | Prevented leakage-driven overfitting | Session-CV said 0.534; reality 0.6144. The switch is *why* I climbed #79→#27. |
| **Drop learning-objective TARGET ENCODING** | Fixed below-baseline LB | It adds huge session-CV signal but **zero** on unseen objectives (numeric+enc 0.6019 ≈ numeric 0.5988 obj-grouped); removing it: LB 0.6224 → 0.6144. |
| **Keep objective-TEXT TF-IDF** | Generalizable topic difficulty | Dropping it (v2) worsened obj-CV 0.585 → 0.606. |
| **Enriched recency/dynamics features** (last-quarter talk, first→second-half trajectory, response latency, tutor-run length) | obj-CV **AUC 0.631 → 0.648** | Full-model objective-grouped CV. |
| **Literature-grounded features** (talk-moves: pressing-for-reasoning / revoicing / eliciting; self-explanation; tutor uptake; ICAP constructive-vs-passive) | **+0.0102 AUC, +0.0023 logloss** | Numeric-only obj-grouped A/B: 0.5909 → 0.6011 AUC. Tutor-agnostic ⇒ expected to transfer better. |
| **Domain-adaptive MLM warmup on tutoring corpus** (MathDial + our transcripts) | **+0.0088 AUC, +0.0037 logloss** | BERT-mini obj-grouped A/B (18k subset): baseline 0.6171 → adapted **0.6259** AUC; logloss 0.6146 → 0.6109. External-data (MathDial) pretraining transfers. |
| **Classical + transformer ENSEMBLE** | **+0.0080 AUC, +0.0049 logloss** | obj-grouped 3597 held-out: classical 0.6318 → blend **0.6398** (w_transformer=0.33); prediction corr 0.65 ⇒ genuinely decorrelated. Implemented in the A100 container-trainer. |

## ❌ What HURT (or didn't help)
| Measure | Effect | Evidence |
|---|---|---|
| **Learning-objective target encoding** | LB **worse than baseline** | id-1002 = 0.6224 > 0.6088. Classic CV leakage. |
| **Over-shrinkage** (shrink toward prior too much) | LB got worse | id-1022 (a=0.12)=0.6200 vs id-1019 (a=0.4)=0.6144. Convexity ⇒ optimum is a≥0.4; the model is **well-calibrated at higher confidence**. My initial "shift = overconfidence" assumption was **wrong** (that was a *feature* problem, not calibration). |
| **Heavier LR regularization** (C=0.3 vs 1.0) | obj-CV 0.585 → 0.606 | Over-regularization removed signal. |
| **"v2" config** (drop LO-tfidf, add student-tfidf, C=0.3) | obj-CV worse (0.606) | Composite of the two rows above. |
| **Local deep learning** (ELECTRA-small, 384-token recency, on M1) | AUROC **0.58 < 0.648** classical | A tiny truncated transformer can't match full-transcript TF-IDF + engineered features; the M1 (8 GB, no CUDA) can't train a real one. |
| **v3 catalog features** (in-session correctness *proxy*, Hattie feedback levels, telling, objective difficulty, content coverage) | **−0.008 to −0.016 AUC → REVERTED** | Numeric obj-grouped: base 0.6011 → +refined 0.5929 → +full 0.5854. Two causes: (a) **objective-derived** features (difficulty/coverage) can't transfer to *unseen* objectives (same lesson as LO target-encoding); (b) the correctness-proxy is redundant with existing praise/affirmation features and its noise swamps the one strong sub-signal (`proxy_last` corr 0.137 alone). **Lesson: not every theory-grounded idea transfers — the *behavioral talk-move* features did, this batch did not.** |
| **Running two heavy jobs at once on 8 GB** | Both crawl (swap-thrash) | Process lesson: strictly sequential heavy jobs. |

## ⚪ Neutral / built-but-unvalidated
| Measure | Status |
|---|---|
| Probability **clipping** | No gain (blend already conservative). |
| **A100 container-trainer** (ModernBERT fine-tuned in-container, ensembled, classical fallback) | Built + locally validated (code); ModernBERT-on-A100 accuracy unvalidated (smoke stuck on 692 MB upload; fallback protects). |
| **Frozen sentence embeddings** | Too slow to finish on M1; inconclusive. |
| **Diverse-model ensemble** | Tooling built; assembly pending. |

---

## Methodology lessons (durable)
1. **Group your CV by the thing the test holds out.** Here that's learning
   objectives, not sessions. A leaky split cost me a submission.
2. **Don't conflate a feature problem with a calibration problem.** id-1002's bad
   score was leaky features, not overconfidence — I mis-generalized and wasted a
   submission shrinking a well-calibrated model.
3. **The leaderboard keeps your best score**, so late experiments have zero
   downside — but you still only get 3/week, so validate leakage-free first.
4. **Theory-grounded, tutor-agnostic features transfer better** than
   train-specific vocabulary (the measured weakness: obj-CV 0.648 → LB 0.604).
5. **8 GB is the binding constraint** — smaller models, strictly sequential.

## Pending (to append as results land)
- Domain-adaptive pretraining A/B verdict (BERT-mini, MathDial+transcripts).
- Full-model objective-grouped re-check with literature features.
- Final robust ensemble (classical-with-lit + best transformer) objective-grouped.



===== FILE: docs/MODEL_ARCHITECTURE.md =====
# Model-architecture analysis: is a shallow model the right fit for these insights?

*Requested exercise: evaluate whether the current best model type best leverages
the literature insights, and what other AI model types could leverage them more —
to compete with or combine with.*

## TL;DR
My current best model is **TF-IDF + gradient boosting / logistic regression** — a
**shallow, bag-of-features** learner. Tonight produced **direct empirical evidence
that this model type is the bottleneck for the deepest insights**, not the features
themselves. The literature's richest signals are **sequential/semantic**; a shallow
model can only consume them as lossy scalar aggregates, and when I did exactly that
it **hurt** (v3 proxy-correctness, −0.008 AUC). The **same underlying signal, given
to a sequence/semantic model (a domain-adapted transformer), *helped* (+0.0088
AUC).** Conclusion: **combine**, don't replace — keep the shallow model as the
robust anchor and add sequence/semantic models that can actually use the structure.

## The core mismatch (with evidence)
The catalog's top themes are all **order-dependent**:
- in-session **correctness trajectory** (all KT methods — BKT/DKT/AKT — are this),
- talk-move **transitions** (bigrams), not counts,
- **confusion → resolution** dynamics,
- **contingency**: tutor move conditioned on the *running* student state (an
  interaction across the sequence).

A bag-of-features model must **collapse these to scalars** (mean, last, streak,
counts), which throws away the temporal information — and adds noise.

**The v3 experiment is the proof.** I encoded the #1 insight (in-session
correctness) as scalar aggregates and fed the GBM: it **hurt** (base 0.6011 →
0.5854 AUC). Yet the identical underlying signal, read *in order* by a transformer
(BERT-mini, MathDial-domain-adapted), **helped** (0.6171 → 0.6259 AUC). Same
information; opposite outcome — **the difference is the model's ability to use
sequence and semantics.** (`proxy_last` alone correlated 0.137 with the target, so
the signal is real; the shallow model just can't exploit it without overfitting on
its noisy scalarization.)

There is also a **semantic** gap: the "Correct-Answer Trap" insight says a correct
*final answer* can mask flawed reasoning. A **lexical** proxy ("tutor said
correct") can't tell these apart — which is partly why v3 failed. Only a model that
*reads meaning* (transformer / LLM) can.

## Model types that would leverage the insights MORE
| Model type | Why it fits the insights | Evidence / status | Risk |
|---|---|---|---|
| **Transformer over raw dialogue** (ModernBERT/BERT) | Reads turns *in order* → captures talk-moves-in-context, in-session correctness, reasoning quality *implicitly*; domain-adaptable on tutoring corpora | **Validated**: +0.0088 AUC from MathDial adaptation; decorrelated from classical | Overfits/transfer risk; needs GPU (A100 container) |
| **Knowledge-Tracing sequence model (DKT/AKT)** | *Purpose-built* for "predict next-correct from a sequence of attempts"; models the mastery **trajectory** the shallow model can't | Not yet built; catalog's most-recurring theme | Needs reliable per-turn labels; in-session attempts are sparse/noisy here |
| **Hierarchical utterance→session encoder** | Encode each utterance (move + content) → attention over utterance embeddings → local moves *and* global trajectory; efficient on long transcripts | Not built | More engineering |
| **LLM-as-extractor / judge** (vLLM on the A100) | Rates *reasoning quality* & *semantic* in-session correctness — exactly the "Correct-Answer Trap" gap that sank the lexical proxy | Not built; runtime *has* vLLM | Cost/calibration; unvalidatable locally |
| Graph/relational over dialogue moves | Models references/contingency as edges | Exotic | Low ROI now |

## Compete or combine?
**Combine.** They have complementary error profiles:
- The **shallow classical** is my measured *robustness* strength — it transfers
  under the train→test shift better than fragile deep signals, and it's cheap.
- The **transformer / KT / LLM** capture the *sequential + semantic* structure the
  shallow model cannot.

Two ways to combine (both recommended):
1. **Ensemble** (weighted by held-out performance) — already implemented in the
   A100 container-trainer (classical + ModernBERT, weight chosen on an
   objective-grouped hold-out, classical fallback). My data shows the transformer
   is **decorrelated**, so this is a genuine robustness+accuracy gain.
2. **Deep model as a *feature generator* for the shallow model** — e.g., an
   LLM-extracted "understanding score" or a KT "mastery estimate" becomes **one
   semantic feature** in the classical model. This is the fix for why v3 failed:
   replace the **lexical** correctness proxy (which hurt) with a **semantic** one
   from a model that reads meaning. Best of both: robustness of trees + semantics
   of the LLM.

## Recommendation (ranked, for the next window)
1. **Scale the transformer+classical ensemble** (container-trainer, A100) — the
   only validated "better model type"; already built.
2. **LLM-as-extractor** for a semantic in-session-correctness / reasoning-quality
   score → feed as a feature *and* ensemble. Directly targets the insight that beat
   the lexical proxy. Needs the A100/vLLM; validate via free smoke tests.
3. **KT-style attention model** over a per-turn feature sequence (talk-move +
   engagement + proxy) predicting next-correct — the most elegant match to the
   target; a research bet requiring reliable per-turn features.
4. **Keep the shallow classical** as the robust anchor and ensemble member — it is
   *not* obsolete; it's the generalization backbone.

## The meta-lesson
Tonight's most important architectural finding isn't a score — it's that **the
model type, not just the feature set, gates the deepest insights.** The sequential
correctness signal *hurt* a shallow model and *helped* a sequence model. That single
contrast is the strongest argument for investing in sequence/semantic architectures
(transformer now, LLM-extractor and KT next), combined with — not replacing — the
robust shallow anchor.



===== FILE: docs/RESULTS_AND_STRATEGY.md =====
# Results, Learnings & Next-Window Strategy

## Leaderboard journey (competition: Trace the Ace, 329 participants)

| Submission | Model | shrink_a | Public Log Loss | Rank |
|---|---|---|---|---|
| id-1002 | TF-IDF + LO **target-encoding** (leaky) + numeric | 1.0 | 0.6224 | #79 |
| **id-1019** | Enriched feats, no LO-enc, keep LO-tfidf | 0.4 | **0.6144** | **#27** |
| id-1022 | same model | 0.12 | 0.6200 | #27 (kept best) |

**Net result: #79 → #27 (+52 places). Best public score 0.6144.**
Context: leaderboard #1 = 0.6013; constant-baseline ≈ 0.6088. The task is
intrinsically near-noise (the entire field sits within ~0.008 of baseline; top
AUROC ≈ 0.63).

## What worked
1. **Found the CV→leaderboard leak.** The first model's session-grouped CV said
   0.534, but the learning-objective **target encoding** was leakage: it adds a
   huge boost when objectives are shared across folds and **nothing** when
   objectives are unseen (objective-grouped CV: numeric+enc 0.6019 ≈ numeric-only
   0.5988). The public test has largely unseen objectives → that "signal"
   vanished and the model scored *below* baseline (0.6224).
2. **Switched validation to objective-grouped CV** (hold out learning objectives)
   — a realistic, leakage-free estimate.
3. **Enriched features** (recency/dynamics: last-quarter student talk, praise,
   uncertainty; first→second-half trajectory; response latency; tutor-run length)
   + student & objective TF-IDF raised objective-grouped **AUC 0.631 → 0.648**,
   and dropped the leaky encoding. This is what moved #79 → #27.

## The calibration mistake (and the lesson)
I assumed the train→test shift caused **overconfidence**, so I shrank predictions
toward the base rate (`p → a·p + (1−a)·0.7025`). The three anchors proved the
**opposite** for the good-feature model:

```
shrink_a:   0.12   0.40      (same model)
LogLoss:  0.6200 0.6144      <- LESS shrinkage is BETTER
```

Log-loss-vs-shrinkage is convex, so the minimum is at **a ≥ 0.4** — the model is
**well-calibrated at full confidence**, and shrinking *discarded* signal. I
mis-generalized from id-1002 (whose bad score was caused by *leaky features*, not
overconfidence) and spent the last submission (a=0.12) testing the wrong
direction. **Lesson: never assume the shift direction — measure it; and don't
conflate a feature problem with a calibration problem.**

Estimated: the **unshrunk** version of the id-1019 model would score ~0.607–0.61
(≈ top 10–15). It is prepared and ready (see below).

## Deep learning attempt
The container offers an A100 + vLLM, but the local machine is an **8 GB M1** — too
small to fine-tune real transformers, and embedding 35k long transcripts took
hours. A small ELECTRA-small fine-tune (feasible locally) reached only **AUROC
0.58** objective-grouped (< classical 0.648) — a 14M model on a truncated
384-token window can't match full-transcript TF-IDF + engineered features.
Deep learning is not the lever here without container-side (A100) training,
which cannot be validated locally.

## Next-window strategy (submissions reset **2026-07-14 UTC**, 3/week)
`submission.zip` is prepared with the **unshrunk** model (`shrink_a = 1.0`).
To find the true optimum, use the 3 submissions to bracket shrink_a around the
convex minimum (which is ≥ 0.4):

1. **`shrink_a = 1.0`** (prepared)  — the natural, full-confidence model.
2. **`shrink_a = 0.7`** — in case the optimum is interior.
3. **`shrink_a ∈ {1.3}`** (mild *sharpening*, `p→clip(a·(p−π)+π)`) — only if 1.0
   still beats 0.7, indicating the min is at/after full confidence.

Changing shrink_a needs **no retraining** — it's a post-hoc scalar in the model
config. Patch + repackage:
```bash
python - <<'PY'
import joblib; p="submission/assets/artifacts.pkl"; a=joblib.load(p)
a["cfg"]["shrink_a"]=1.0; joblib.dump(a,p,compress=3)
PY
python automation/package.py
python automation/submit_code_v2.py submission.zip normal "note"
```

## Ideas to raise the ceiling (would need the next window to validate)
- My leaderboard AUROC (0.604) < leaders (0.63): the transcript-vocabulary TF-IDF
  likely overfits to train tutors and transfers poorly. A model leaning more on
  **objective/topic difficulty + robust pedagogical numerics** (provider-agnostic)
  and less on raw vocabulary may transfer better.
- **Ensembling** decorrelated models is robust to distribution shift; even a weak
  transformer, if bundled and A100-trained, could help an ensemble.
- **A100 container-training**: bundle the train data + a fine-tune script and
  train a real (full-context) transformer inside the 6-hour container — the only
  way to get GPU training here. Validate the code via free smoke tests.



===== FILE: literature/INSIGHTS.md =====
# Literature Study — Insights Catalog

Synthesized from **317 ideas** extracted across **290 harvested sources** (16-cluster harvest + 21-agent deep-read + synthesis). Each item is transcript-computable for predicting next-answer correctness. Sources in [`papers/`](papers/) & [`papers2/`](papers2/); raw list in [`harvest_sources.json`](harvest_sources.json); catalog in [`INSIGHTS_catalog.json`](INSIGHTS_catalog.json).

## Recurring themes

- Reconstruct a per-attempt correctness proxy from the tutor's confirm-vs-correct reaction to each student answer, then derive recency/streak/two-count/latent-mastery features from that sequence. This is the single most-recurring idea across the corpus (DKT, PFA, BKT, RPFA, Scarlatos KT-in-dialogue all reduce to it) and the closest label-free analogue of the target.
- Anchor features to the SPECIFIC upcoming concept/item, not global session aggregates: concept-recency lag (time/turns since the assessed terms were last touched), practice count on the target KC, and how much the session actually covered the question's content.
- Item difficulty is a required covariate: near-noise correctness is largely mastery x item-difficulty, so surface-complexity of the upcoming question text (readability, symbolic density, rare-word ratio, step count) is a driver none of the current transcript-side features capture.
- Feedback-quality typology beyond praise/corrective keyword presence: Hattie levels (task/process/self-regulation/person), person-vs-task praise direction, post-error elaboration, and mistake pinpoint/actionability.
- Contingency and moderation as INTERACTIONS rather than main-effect counts: tutor move x running student mastery, support-after-error minus support-after-success, early-ability x discourse move, elaboration x participation.
- Sequence/order over bag-of-counts: move-transition bigrams, trailing streaks, first->second-half trajectories, and soft exponential recency kernels replacing hard last-quarter windows.
- Distinguish productive from unproductive struggle: confusion resolution vs persistence, wheel-spinning / unresolved-error streaks, self-correction/self-monitoring, and executive vs instrumental help-seeking.
- Tutor-agnostic normalization for transfer: within-session/within-tutor z-scores, student:tutor ratio forms, and cohort percentile-ranks so verbosity/style baselines cancel.
- Student-side depth-of-processing lexical signals: disfluency, specificity, evidence-grounding, content novelty, initiative, argument completeness (claim+warrant), and challenge-with-justification.
- Answer leakage / tutor 'telling' as a first-order confound on the label itself: revealing the answer inflates next-correct without learning and changes what the target measures.
- Near-noise pipeline discipline: strict prefix-only (leakage-safe) feature construction, session-grouped CV, probability calibration with abstention, coefficient shrinkage sized to the small true effect, and a feature-reliability gate.
- Affect/confusion dynamics are directional, not level: valence trajectory (recovering vs deteriorating), frustration/giving-up markers, and resolved-then-correct episodes.
- Optional heavier semantic layers with high potential value but real effort/reliability caveats: sentence-embedding pooling + semantic uptake, and a frozen LLM role-playing the student to estimate P(next correct) or scoring a discourse rubric / knowledge-state summary.

## Ranked catalog

| # | Feature/method | Value | Effort | Status | How to compute (short) |
|---|---|---|---|---|---|
| 1 | proxy_correctness_history | high | structural | **IMPLEMENTED (v3)** | Walk turns in timestamp order. For each student turn that answers a preceding tutor question, read the immediately-following tutor turn: label the answer 1 if i… |
| 2 | concept_recency_lag | high | structural | candidate | Tokenize the upcoming question stem (or its stated learning-objective text) into lemmatized, stopword-stripped content terms. Scan backward through prior turns … |
| 3 | upcoming_question_content_coverage | medium | structural | **IMPLEMENTED (v3)** | Extract content terms/keyphrases from the upcoming question (lemmatized, stopword-removed). Coverage = fraction appearing anywhere in the transcript; also a stu… |
| 4 | upcoming_question_difficulty_proxy | medium | lexical | **IMPLEMENTED (v3)** | From the next question's text compute: token count; words-per-sentence; mean word length / syllables-per-word (Flesch-Kincaid); symbolic density = (digits + mat… |
| 5 | telling_vs_eliciting_and_answer_leakage | medium | structural | **IMPLEMENTED (v3)** | Classify each tutor turn: telling = declarative/imperative answer or full-procedure statement (the answer is/it'?s/you get/so it equals/= <value>/multiply then.… |
| 6 | unresolved_struggle_streak | medium | structural | **IMPLEMENTED (v3)** | Segment into problem episodes at new-problem cues (tutor turn with next/new problem/let'?s try/okay so, or a topic shift = low content-word overlap with the pri… |
| 7 | feedback_level_composition | medium | lexical | **IMPLEMENTED (v3)** | Classify each tutor turn by lexicon into: task/product (correct/the answer is/that'?s wrong); process/strategy (the strategy/try thinking about/the reason is/me… |
| 8 | post_error_feedback_quality | medium | structural | candidate | For tutor turn(s) immediately after a proxy-incorrect student answer compute: word count (elaboration length); explanation connective present (because/so/which … |
| 9 | person_vs_task_praise_ratio | low | lexical | candidate | Within tutor praise turns, split person/ego-directed (praise token + 2nd-person pronoun / trait adjective, no task referent: you're smart/good boy/you're great)… |
| 10 | tutor_contingency | medium | structural | candidate | Using proxy-correctness labels: (a) support_after_error - support_after_success, where support = tutor-turn length + count of scaffolding/hint markers on the fo… |
| 11 | move_by_mastery_interactions | medium | structural | candidate | Set proxy_mastery = running share of proxy-correct attempts up to each point. Build interaction features: press_density x proxy_mastery, revoice_density x (1-pr… |
| 12 | feedback_delay | medium | structural | candidate | For each student answer turn, measure intervening turns AND elapsed seconds until the next tutor turn carrying evaluative/corrective content. Aggregate mean/med… |
| 13 | student_initiative_and_novelty | medium | structural | candidate | For each student turn label 'initiating' if it opens with a question, introduces on-topic content words absent from the immediately preceding N tutor turns (nov… |
| 14 | confusion_resolution_and_monitoring_shift | low | structural | candidate | Detect confusion onset in student turns (confused/don'?t get/understand/lost/huh/repeated hedges/'?'). Mark RESOLVED if within K turns a student turn is proxy-c… |
| 15 | student_self_correction_markers | low | lexical | candidate | Regex over student turns for self-initiated repair/monitoring: wait/no wait/actually/I mean/let me redo/recheck/try again/I made a mistake/oh I see/that'?s not … |
| 16 | help_seeking_and_formative_initiation | low | lexical | candidate | Executive help cues in student turns (just tell me/what'?s the answer/give me the answer/I give up/can you just do it) vs instrumental (why/how do i/can you exp… |
| 17 | rapid_guess_bursts | low | structural | **IMPLEMENTED (v3)** | Flag a student answer turn glib-fast if response latency (timestamp delta from the prior tutor turn) is below the student's own session median AND the turn has … |
| 18 | student_disfluency_density | low | lexical | candidate | Regex over student turns for filled pauses (\bu[mh]+\b, \ber+\b, hmm), immediate word repetitions (\b(\w+)\s+\1\b), self-repair (I mean/no wait/or rather/--), t… |
| 19 | affect_valence_trajectory | low | lexical | candidate | Score each student turn valence = positive markers (got it/makes sense/oh nice/I see) minus frustration/negative markers (ugh/this is hard/stuck/confusing/give … |
| 20 | move_transition_and_act_entropy | low | lexical | candidate | Tag each turn with a coarse move/act via lexical+punctuation rules (question='?', tell, revoice/restate, praise, correct, backchannel, directive). Emit normaliz… |
| 21 | question_cognitive_level | low | lexical | candidate | Classify each interrogative turn: higher-order/authentic (why/how/explain/justify/what if/compare/predict/what do you think/how could/why might) vs recall/close… |
| 22 | question_chain_and_step_granularity | low | structural | candidate | Find runs of tutor question turns separated only by short student responses within one episode -> mean/max tutor question-chain length and count of chains >=2 e… |
| 23 | student_content_density_and_grounding | low | lexical | candidate | Per student turn: content density = (domain/content tokens: numerals, operation words, learning-objective vocab, contentful nouns/verbs - backchannel/filler ok/… |
| 24 | complete_argument_and_challenge_justify | low | lexical | candidate | Count student turns with claim+warrant co-occurrence (a declarative answer clause AND a justification clause because/so/therefore/which means/that'?s why in the… |
| 25 | multi_turn_coherence | low | structural | candidate | For each turn t compute content-lemma Jaccard between t and the union of turns t-1..t-3 (stopwords removed); average across the session and take its first->seco… |
| 26 | worked_example_and_generation_first | low | structural | candidate | Flag tutor worked-example turns (long, >=2 sequence markers + a result, no preceding student attempt on that content). Flag student independent-attempt turns (m… |
| 27 | interactive_coconstruction | low | structural | candidate | Tag a student turn Interactive if it (a) has high lexical overlap with the immediately-preceding tutor turn (uptake) AND (b) adds new content words not in that … |
| 28 | within_session_and_cohort_normalization | medium | structural | candidate | For every count/rate feature add: its within-session z-score or rank; its student:tutor ratio form (so absolute chattiness cancels); and its cohort percentile-r… |
| 29 | bkt_running_mastery_filter | medium | structural | candidate | Cluster answer contexts into a few 'skills' by TF-IDF of learning-objective terms. Run a fixed-parameter 2-state BKT/HMM (learn/guess/slip ~0.3/0.2/0.1) over ea… |
| 30 | leakage_safe_prefix_and_grouped_cv | medium | structural | candidate | Build every feature for predicting item t from a strict prefix (turns strictly before the assessment) and audit that no feature reads the answer turn or later t… |
| 31 | exp_decay_recency_and_reliability_gate | low | structural | candidate | For any per-turn scalar x_i (uncertainty, affirmation, latency, valence) compute a soft decayed aggregate sum(w_i*x_i)/sum(w_i) with w_i=exp(-lambda*(t_last - t… |
| 32 | sbert_semantic_embeddings | medium | sequence-model | candidate | Embed each turn with a frozen sentence encoder; aggregate the last-k student turns and (separately) tutor turns by element-wise mean/std/min/max pooling into a … |
| 33 | llm_simulated_student_and_rubric | high | llm | candidate | Prompt a frozen instruction-tuned LLM with the role-tagged transcript prefix plus the upcoming question, asking it to role-play the student and emit either an a… |

## Implemented this pass (v3), measured objective-grouped
- In-session **correctness proxy** (#1) + recency/streak/PFA stats — `proxy_last` corr with target = **0.137** (strongest single feature).
- **Feedback levels** (Hattie #7), **telling-vs-eliciting** (#5), **objective difficulty** (#4), **content coverage** (#3), **rapid-guess latency** (#17).

See [../docs/EXPERIMENT_LOG.md](../docs/EXPERIMENT_LOG.md) for the measured helped/hurt of each batch.


===== DATA SCHEMA & STATS =====
train_features.csv cols=['response_id', 'session_id', 'learning_objective_id', 'learning_objective'] rows=35072
train_labels.csv cols=['response_id', 'is_correct'] label mean=0.7025
n_sessions=22821 n_objectives=398
responses/session: mean=1.54 max=10
transcript cols=[session_id,utterance_id,role,content,timestamp]; roles=tutor/student/background; timestamps elapsed HH:MM:SS; utterances/session mean~258


===== SAMPLE TRANSCRIPTS (3, real) =====

--- session aaaedit (label is_correct=[0.0, 1.0]) first 25 turns ---
[00:00:00] tutor: Hello?
[00:00:00] background: [unclear]
[00:00:01] background: Hello.
[00:00:01] tutor: Hi, Lachlan. How are you doing today?
[00:00:07] student: Good.
[00:00:08] tutor: Okay, that's great to know. So, how was the weekend?
[00:00:12] student: Good.
[00:00:16] tutor: That's wonderful. What did you do during the weekend?
[00:00:21] student: During the weekend, on the 15th of February start, I've been just been trying to [UNCLEAR]. I've been to the swimming pool there. Then I've done some 
[00:00:53] tutor: Okay, so that's great. So I'm so glad to hear that you've been doing great and Had a good rest, I hope, during the weekend?
[00:01:03] student: Yeah.
[00:01:04] tutor: Okay, so as usual, today also we are going to do more new questions together. So feel free to ask any questions related to the lesson. I'm always here
[00:01:36] student: Yeah, I know.
[00:01:38] tutor: Yeah, so what do you already remember about this lesson?
[00:01:43] student: Oh, where you started. Choosing a mental or written method, it says one of that.
[00:01:47] tutor: Mm-hmm. Yeah, so when we are given a question, we can either do it mentally depending on the calculation, or we can do a written method, isn't it?
[00:02:01] tutor: So today we are going to start from where we left off. So that was great how we were able to get to the answer quickly. So good job on it. So sorry. S
[00:02:19] student: Yeah.
[00:02:21] tutor: So we are going to start from the question— C. Let's start from question C. Yeah, yeah. Okay, so which method are we going to use? Are we going to use
[00:02:37] student: I already just said it. I already said it.
[00:02:41] tutor: Okay, all right, so we'll go with that.
[00:03:09] tutor: So, from which column do we start when we do a column method? [unclear] [unclear]
[00:03:31] tutor: Okay, there you go.
[00:03:43] tutor: So for the Question D, shall we use a mental method?
[00:03:50] student: No, written method.

--- session aaaptjd (label is_correct=[0.0]) first 25 turns ---
[00:00:00] background: [unclear]
[00:00:00] tutor: Hello?
[00:00:04] student: Hello?
[00:00:08] tutor: Hi, how are you doing today?
[00:00:09] student: I'm good.
[00:00:11] tutor: Okay, I really like how you're always on time, by the way. Exactly on the clock, you're always on time. That's really cool. I just wanted to commend y
[00:00:26] student: Adding and subtracting amounts of money to give change.
[00:00:30] tutor: Uh-huh. And let's try the first question.
[00:00:33] student: Selma buys a pencil case costing £2.65. She pays with a £5 note. How much change does she get?
[00:00:47] student: Uh, okay. [unclear]
[00:01:18] student: Is this right?
[00:01:19] tutor: Okay, let's go ahead and check your answer, okay? So how much did someone buy the pencil case for?
[00:01:25] student: £2.65p. [Speaker:Tutor]...with? [Speaker:Beatrix] A £5 note.
[00:01:32] background: Is it possible for us to have 95p left?
[00:01:40] student: Oh, [UNCLEAR]. Oh, 35. [Speaker:Tutor] 35, okay. That's the correct answer. Can you tell me how you worked it out?
[00:01:47] student: So first with the 65p, I worked the 65p out first. So what I did is I counted from 60 to 90 because it can't get to 100. So 60, 70, 80, 90, which woul
[00:02:37] student: 65 pence.
[00:02:39] tutor: No, how many pounds did you have? So with the £2, with the 65, with the 35.
[00:02:46] student: That would be £5. [unclear]
[00:02:50] student: Oh no, sorry, that would be £3.
[00:02:55] tutor: £3. So we only have £2 left to work out, right? That would give us £2. OK, I like how you worked that out. I think you used this method last week as w
[00:03:17] student: Mo buys a comic for £2.25. He pays with a £5 note. How much change will he receive?
[00:03:27] student: So— oh wait, should I explain how to do it while I'm working it out?
[00:03:33] tutor: Yes, that would be much easier so I don't have to ask you at the end.
[00:03:37] student: OK, so 25 would get into 90, so if I was to put the 2, how much would the 2 get into 90? 9. So there would be 2, 3, 4, 5, 6, 7, 8, 9, so 7. And then I

--- session aabkeov (label is_correct=[1.0]) first 25 turns ---
[00:00:00] student: Hello.
[00:00:00] background: [unclear]
[00:00:01] tutor: Hello.
[00:00:03] student: Hi.
[00:00:04] tutor: Welcome back.
[00:00:05] student: Thank you.
[00:00:07] tutor: Hi, Kaelan. [unclear]
[00:00:13] student: Hello?
[00:00:15] tutor: Hi.
[00:00:16] student: Yeah, how are you doing today?
[00:00:18] tutor: I'm good, how are you?
[00:00:20] student: I'm good too, thank you for asking. So I will be your tutor for today. Can you confirm on how can I call you? Is it Kaelan?
[00:00:29] student: Yeah, it's going a little bit.
[00:00:32] tutor: Okay, great, Kaelan. So tell me about your day at school today. What did you study? I mean, what did you learn? [unclear]
[00:00:41] student: Science.
[00:00:43] tutor: Okay, how did it go? Did you enjoy it?
[00:00:44] student: It's all right. Yeah, it's all right.
[00:00:47] tutor: Okay, great. Yeah, so now let's move on with our lesson for today, which is we are going to continue with working with ratio. Yep. Yeah. All right, Ka
[00:01:05] student: No worries. Do you remember trying this question?
[00:01:11] background: Yeah.
[00:01:13] tutor: Great. So what does ratio mean simply?
[00:01:20] background: Ratio means—
[00:01:26] student: Uh-huh. [unclear]
[00:01:30] tutor: Okay, basically it's like we go with this, yeah? Like when we compare two numbers or two members and we see how much is to the other member, isn't it?
[00:01:49] student: [unclear]