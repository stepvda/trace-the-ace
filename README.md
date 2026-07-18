# Trace the Ace — Tutoring Outcomes Prediction

Solution for the DrivenData / K-12 AI Infrastructure competition
[*Trace the Ace*](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/).

> ### 👀 Reviewing this project to suggest improvements? Start with **[docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md)**.
> It is a self-contained brief: the problem, the current model & score, the ideas already
> ruled out (so you don't re-suggest them), the open opportunities, and the hard rules.

**Task.** Given a student–tutor session transcript and a short learning-objective
description, predict the probability that the student answers the *next* quiz question on
that objective correctly (`is_correct` ∈ {0,1}).

**Metric.** Log loss (binary cross-entropy); AUROC for reference. Note: the often-cited
"constant baseline 0.6088" is the entropy of the *train* rate and was **never observed on
the leaderboard** — the true constant is ≈ **0.6236** (test base rate ≈ 0.685), so this
model *beats* the constant. See [docs/REVIEW_GUIDE.md §4](docs/REVIEW_GUIDE.md) for the
calibration analysis.

**Submission type.** Code execution: a `submission.zip` with `main.py` at the root +
`assets/`, run **offline in a container** (Python 3.12, A100 GPU, 6 h, no internet),
reading `data/test_features.csv` + `data/test_transcripts/` and writing `submission.csv`.
3 scored submissions/week; the leaderboard keeps your best.

## Result

**Rank #18 / ~429, best public Log Loss 0.6087** (up from #79 → #27 → #18). The field is
near-noise: it spans only ~0.02 log loss; #1 = 0.6013. Journey and analysis in
**[docs/RESULTS_AND_STRATEGY.md](docs/RESULTS_AND_STRATEGY.md)**.

Two submission artifacts:
- **`submission_classical.zip`** (≈48 MB) — the classical model, calibrated
  (`p → 0.685 + 0.68·(p − 0.7025)`). Entrypoint `submission/main.py`. **This is the 0.6087
  model.**
- **`submission_container.zip`** (≈700 MB, committed as 45 MB split parts — rebuild with
  `./reassemble_container.sh`) — the **in-container ModernBERT trainer** (old design):
  fine-tunes ModernBERT at submission time and ensembles it with the classical model,
  self-gated with a classical fallback. Entrypoint `submission/main_container.py`.
  **Now known to have scored classical-only on every run:** ModernBERT's `sdpa` attention
  emits NaN logits on *padded* batches, so the transformer trained on NaN and the gate
  silently fell back to classical each time — the transformer leg has contributed **zero**
  for the whole competition. This design is being **replaced** (see below); root-cause
  writeup in **[docs/CONTAINER_TRAINER.md](docs/CONTAINER_TRAINER.md)**.

**Transformer leg (in development, not yet submitted).** Fixing the attention bug
(`attn_implementation="flash_attention_2"`) makes ModernBERT-base train properly, and it is
*strong*: on an objective-grouped holdout it reaches ~**0.63 LB-equivalent AUROC** vs the
classical **0.604** — a real discrimination gain on a task where #1 ≈ 0.63. The design has
pivoted from fine-tuning *inside* the container to **pre-fine-tuning on a rented GPU and
bundling the finished weights into an inference-only container** (validated on local OOF
before upload, with a batch-size-1 SDPA fallback and a classical last-resort). This is
OOF-gated and unsubmitted, so **every leaderboard point to date is classical-only.**

## Approach

Pure `scikit-learn` (guaranteed in the offline runtime) blending two complementary
models, validated with **StratifiedGroupKFold grouped by `learning_objective_id`** — the
hidden test set holds out *objectives*, and objective-grouped CV is the leakage-free
estimate (a session/random split leaks; see the guide). Note CV still over-estimates the
leaderboard because of the train→test distribution shift.

1. **Feature engineering** (`solution/features.py`, shared by train & inference):
   - Transcript structure & timing: utterance/word counts, tutor/student balance, turn
     switching, duration/gaps from the `HH:MM:SS` timestamps, response latency.
   - Pedagogical/behavioral signals: talk-moves (pressing-for-reasoning, revoicing,
     eliciting), praise vs. corrective language, student uncertainty vs. affirmation,
     question rates, recency/last-quarter dynamics.
   - Text: TF-IDF over the full transcript, the student-only turns, and the
     objective text; reduced with TruncatedSVD.
   - **No learning-objective target-encoding** — it is leakage (huge in random CV, zero on
     unseen objectives, *worse* on the leaderboard).
2. **Models:** LogisticRegression on TF-IDF + numeric, blended 0.55 with
   HistGradientBoosting on numeric + SVD.
3. **Calibration:** affine recenter onto the estimated test base rate,
   `p_final = 0.685 + 0.68·(p_raw − 0.7025)` (this alone moved #27 → #18).

## Layout

```
data/                         competition data (git-ignored — not in repo)
solution/
  features.py                 feature engineering (train + inference)
  model.py                    fit_pipeline / predict_pipeline + calibration
  dl_common.py                transformer text/representation builder (focused rep + history)
  gpu_*.py, blend_gate.py     rented-GPU ModernBERT harness (A/B, DAPT, 5-fold OOF, ship-gate)
  dl_train.py                 old in-container trainer (being replaced by bundled inference)
  *.py                        experiment scripts (shift_proxy, transfer_ablation, ...)
submission/
  main.py                     classical inference entrypoint (the 0.6087 model)
  main_container.py           A100 ModernBERT-ensemble entrypoint
  assets/                     bundled fitted artifacts (git-ignored — large)
automation/                   Playwright/CDP scripts driving the logged-in Edge browser
  package.py                  build submission_{classical,container}.zip
  upload_big.py               upload >50 MB zips; poll_job.py monitors to score
docs/                         see below
literature/                   ~33-idea catalog synthesized from ~290 education papers
reassemble_container.sh       rebuild submission_container.zip from its committed parts
```

## Reproduce

```bash
pip install -r requirements-lock.txt
# feature/CV/train code lives in solution/ (data must be downloaded into data/ first)
python automation/package.py classical            # -> submission_classical.zip
python automation/test_submission_local.py 800    # end-to-end local runtime check (fallback path)
python automation/package.py container            # -> submission_container.zip (old in-container design)
```

## Docs
- **[docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md)** — **start here if reviewing** — problem,
  current state, ruled-out ideas, open opportunities, constraints.
- [docs/SOLUTION.md](docs/SOLUTION.md) — full methodology, data schema, every feature.
- [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — every measure tried, helped/hurt with numbers.
- [docs/RESULTS_AND_STRATEGY.md](docs/RESULTS_AND_STRATEGY.md) — leaderboard journey & calibration.
- [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) — shallow vs. sequence/semantic analysis.
- [docs/CONTAINER_TRAINER.md](docs/CONTAINER_TRAINER.md) — the A100 transformer-ensemble submission.
- [docs/LLM_EXTRACTOR.md](docs/LLM_EXTRACTOR.md) — the (negative) LLM-as-extractor investigation.
