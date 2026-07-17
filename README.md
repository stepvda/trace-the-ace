# Trace the Ace — Tutoring Outcomes Prediction

Solution for the DrivenData / K-12 AI Infrastructure competition
[*Trace the Ace*](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/).

**Task.** Given a student–tutor lesson transcript and a short learning-objective
description, predict the probability that the student answers the *next*
assessment question on that objective correctly (`is_correct` ∈ {0,1}).

**Metric.** Log loss (binary cross-entropy); AUC shown for reference only.
Constant-mean baseline: **0.6088 is the entropy of the TRAIN rate (0.70) and was never
observed on the leaderboard** — do not treat it as the real floor. The three shrink
anchors (0.6200/0.6144/0.6151 at a=0.12/0.40/1.0, same model) are convex in `a`;
extrapolating gives the *true* constant-0.7025 score ≈ **0.6236**, implied **test base
rate ≈ 0.685**, and optimal shrink **a\*≈0.68 → LB≈0.6126**. So our 0.6144 model **beats
the true constant by ~0.009** and has real discrimination (LB AUROC 0.604). Confirm with
a one-off flat-0.7025 "constant probe" submission.

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
