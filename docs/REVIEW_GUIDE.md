# Reviewer's Guide — Trace the Ace

**Read this first.** It is a self-contained brief for someone reviewing this project
to suggest ways to improve the model. It states the problem, the current solution and
score, the ideas already ruled out (so you don't re-suggest them), the open
opportunities, and the hard constraints. Other docs go deeper; this is the map.

> **What I want from a reviewer:** concrete, mechanism-grounded ideas to lower the
> **public log loss** (currently 0.6087, rank #18) that will actually **transfer to the
> hidden test set** — not ideas that only improve a local cross-validation number.
> The single hardest thing about this problem is that local validation does **not**
> reliably predict the leaderboard (see §3).

---

## 1. The problem

**Competition:** DrivenData "Trace the Ace" (tutoring-outcomes). ~429 teams.

**Task:** Given a **student–tutor session transcript** plus a short **learning-objective
description**, predict the probability that the student then answers the **next quiz
question on that same objective correctly** (`is_correct` ∈ {0,1}).

**Metric:** **Log loss** (binary cross-entropy). Lower is better. AUROC is shown for
reference only but is diagnostically useful (see §4).

**Data:** 35,072 training rows / 22,821 sessions / 398 learning objectives. Base rate
`P(correct) ≈ 0.70`. Each row = one (session, objective) pair; a session with multiple
objectives yields multiple rows that **share the same transcript**. Two tutoring
providers contribute data: **Eedi** (short typed chats) and **Third Space Learning**
(longer voice-transcribed lessons). Columns: `response_id, session_id,
learning_objective_id, learning_objective` (+ the transcript CSV per session).

**Submission:** *Code execution.* You upload a `submission.zip` (with `main.py` at root)
that runs **offline in a container** (A100 80 GB GPU + vLLM available, Python 3.12,
6 h limit, no internet) against the hidden test set. **3 scored submissions/week; the
leaderboard keeps your best.** Free "smoke tests" (100-row sample) validate the code.

**Why it's hard — it is a near-noise task with a distribution shift.** The whole
leaderboard spans only ~0.02 log loss (from #1 = 0.6013 down through the pack). There is
very little extractable signal, and the train→test distribution shifts (see §3), so most
things that help in cross-validation do **not** help on the leaderboard.

---

## 2. The current solution & score

**Best public score: 0.6087, rank #18 / ~429** (was #27 at the start of this work).

**Model (all in `solution/`):**
- **Text:** TF-IDF over three fields — full transcript, student-only turns, and the
  learning-objective text — reduced with TruncatedSVD.
- **Numeric:** 64 hand-built **behavioral** features (turn structure, response latency,
  talk-moves à la Accountable Talk / TalkMoves, praise/uncertainty markers,
  recency/last-quarter dynamics). See `solution/features.py`.
- **Blend:** LogisticRegression (on TF-IDF + numeric) blended 0.55 with
  HistGradientBoosting (on numeric + SVD). Pure sklearn/numpy for container portability.
- **Calibration (this matters — see §4):** an affine recenter applied to the blended
  probability: `p_final = 0.685 + 0.68·(p_raw − 0.7025)`.
- **No learning-objective target-encoding** (it is leakage — see §5).
- **Leaderboard AUROC ≈ 0.604.**

**Container ensemble (built, fallback-safe, `submission/main_container.py`):** the
classical model **plus** a ModernBERT transformer fine-tuned in-container on the A100,
blended by a self-gating weight, with the classical model as fallback. Its last run
**fell back to classical** (the transformer errored — see §6), so its potential is
unrealized. This is the leading open opportunity.

**Reproduce:** `pip install -r requirements-lock.txt`; training/feature code in
`solution/`; `python automation/package.py classical|container` builds the zip;
`automation/test_submission_local.py` runs the container locally (fallback path only —
the GPU DL path is A100-only). The competition data is not in the repo (`data/` is
git-ignored). The container zip is committed as 45 MB split parts — rebuild with
`./reassemble_container.sh`.

---

## 3. The one thing that makes this hard: **local validation doesn't predict the leaderboard**

This is the crux, and every reviewer should internalize it before proposing ideas.

- **Cross-validation systematically over-estimates the leaderboard.** The train→test
  shift (different objectives, different provider mix, different students) is not present
  in any split of the training data, so a model that fits the training distribution well
  can do worse on the shifted test.
- Concretely: the classical model beats a constant by **+0.027 log loss on training
  (objective-grouped CV)** but the shift **attenuates that to ~+0.014 on the real test**.
  AUROC drops from ~0.645 (CV) to 0.604 (leaderboard).
- **The correct CV grouping is by the thing the test holds out.** The test objectives
  are effectively **unseen**, so validation must hold out whole learning objectives
  (`StratifiedGroupKFold` on `learning_objective_id`). A session-grouped or random split
  leaks (see §7 — the CV itself has a subtle leak we only recently found).
- **Practical consequence:** the only fully trustworthy signal is the leaderboard itself
  (3 scored subs/week, best-kept). Ideas that only move a local proxy are suspect.

---

## 4. The calibration insight (avoid a trap that misled this project for a while)

For a long time this project believed its model "scored *below* the constant-mean
baseline of 0.6088" and was therefore extracting negative signal. **That was wrong**, and
understanding why is important context:

- **0.6088 was never an observed leaderboard score.** It is the entropy of the *training*
  base rate (0.70), computed offline — and it silently assumed the *test* base rate is
  also 0.70.
- Three submissions of the *same model* at different shrink strengths (`a = 0.12/0.40/1.0`
  → log loss `0.6200/0.6144/0.6151`) are **convex in `a`**. Extrapolating that curve shows
  a pure-0.7025 constant actually scores **≈ 0.6236**, and implies the **test base rate is
  ≈ 0.685, not 0.70**.
- So the model **beats the true constant by ~0.014**, and has genuine positive
  discrimination (AUROC 0.604). Recentering predictions onto the 0.685 test rate (the
  calibration in §2) is worth a real, verified gain — it moved the score from 0.6144 to
  0.6091, i.e. **#27 → #18 with no new features**, and confirmed the 0.685 rate on the
  real test.
- **Where the remaining gap to #1 lives:** #1 (0.6013) extracts ~0.022 "nats" of
  information from the transcript; this model extracts ~0.014 (≈65%). Roughly **half of
  our original gap to #1 was calibration (now recovered); the other half is
  DISCRIMINATION** — a better transcript representation we don't yet have. That is the
  target for new ideas (§6).

---

## 5. Already tried and **ruled out** — please don't re-suggest these

Each of these was tested and is a dead end *for a specific, evidenced reason*. Details in
`docs/EXPERIMENT_LOG.md` and `docs/LLM_EXTRACTOR.md`.

| Idea | Verdict | Why |
|---|---|---|
| **Learning-objective target-encoding** / per-objective difficulty as a feature | ❌ dead | Leakage: huge in random/session CV, **zero on unseen objectives**. On the leaderboard it scored *worse* (0.6224 vs 0.6144). Test objectives are effectively unseen. |
| **Objective-text → difficulty** (semantic/embedding difficulty of the objective description) | ❌ dead | Objective text does not predict its difficulty (corr 0.047); doesn't transfer to unseen objectives. |
| **LLM-as-extractor** (an instruct LLM reads the transcript and rates the student's mastery) | ❌ dead | Tested up to a **frontier model** (DeepSeek): the verdict was near-zero / slightly *anti*-correlated and added no robust signal over the classical model. |
| **External Eedi data** (bundle Eedi's own public response data for difficulty priors) | ❌ dead | Every public Eedi dataset is **non-commercially licensed**; the competition requires external data under a commercial-OK license. |
| **Transductive / test-time adaptation** (refit TF-IDF on train+test, recenter on the test mean, importance-weighting using test features, BBSE label-shift) | ❌ **rule-illegal** | Competition rules require each test sample be processed **independently** — no feature parameters fitted across test samples, no test-set aggregates. (BBSE was also empirically useless here.) |
| **More hand-built behavioral features** (a 33-idea literature catalog; 9 "dynamics" features; a lexical in-session-correctness proxy) | ❌ exhausted | The 64-feature set already captures the extractable transcript signal; extra behavioral features land in the noise (confirmed three independent times). |
| **KT / GRU sequence model over per-turn features** | ⚪ neutral | Works but adds ~nothing over the classical (a small model on 8 GB can't beat the tuned classical). |
| **Heavier shrinkage toward the prior** | ❌ | Log loss is convex in shrink; the optimum is ~0.68, not heavy shrink. |

> Note: some of these "null" verdicts rest on the leaky CV described in §7. A reviewer who
> wants to challenge one should insist on a **session-AND-objective-grouped** re-test.

---

## 6. Open opportunities — where fresh ideas would help most

Ranked by the project's current best guess at (impact × plausibility).

1. **Get the transformer working (the top-5 lever).** The leaderboard proves a good
   transcript representation exists and is *findable*: **9 of ~50 teams have AUROC ≥ 0.62,
   two of them on a single submission.** With this project's calibration edge (those teams
   are mostly *badly* calibrated), AUROC 0.615–0.622 would map to **0.606–0.604 → top 5**.
   A ModernBERT container ensemble is built but its last run fell back (the transformer
   errored — likely a `transformers`/ModernBERT runtime issue in the container; it has
   never successfully trained on the real hardware). **Concrete questions:** best
   transformer/approach for long tutoring transcripts under distribution shift? How to
   make an in-container fine-tune robust and validate it with only free smoke tests? Is a
   frozen-embedding + linear head more transfer-robust than fine-tuning?

2. **Objective-conditional representation (a genuine blind spot).** The pipeline computes
   **one feature set per session**, but 59% of rows are **multi-objective sessions** where
   nothing distinguishes *which* objective is being asked about — and within such sessions
   the model's ranking is coin-flip (AUROC ≈ 0.49). Features that are functions of
   `(transcript, objective text)` — e.g. how much of the transcript is *about* the queried
   objective, or an objective-centered transcript window fed to the transformer — could
   add discrimination and, being identity-free, should transfer to unseen objectives.

3. **Fix and re-run the measurement instrument.** The objective-grouped CV has a **sibling
   leak** (multi-objective sessions put identical-feature rows in both train and val
   folds). A session-AND-objective-grouped CV would let us honestly re-test the ideas
   currently marked "null" in §5 — some of those verdicts may be artifacts.

4. **Squeeze the last calibration.** Small and mostly done, but: the recenter pivots on
   the train mean (0.7025) while the model's own mean prediction is ~0.7136; a
   train-derived prior fix (~+0.001) is in the container. Per-segment calibration is
   bounded small and needs probe submissions.

---

## 7. Hard constraints (a reviewer must respect these)

- **Each test sample must be processed independently.** No feature parameters fitted
  across the test set, no test-set aggregates (means, counts, refit vocab). This
  **forbids all transductive tricks** — a common source of "clever" but illegal ideas.
- **External data** must be publicly available under a license that permits commercial use
  (rules out the obvious Eedi datasets).
- **Compute:** development machine is an 8 GB M1 (no CUDA) — can't fine-tune transformers
  locally; the only GPU is the offline A100 in the 6-hour submission container.
- **Metric is log loss**, so calibration matters as much as ranking.
- **Local CV over-estimates the leaderboard** (§3) — treat any local-only gain skeptically;
  the leaderboard (3/week, best-kept) is the only ground truth.

---

## 8. Where to read more
- `docs/EXPERIMENT_LOG.md` — every measure tried, with its measured helped/hurt effect.
- `docs/RESULTS_AND_STRATEGY.md` — the leaderboard journey and calibration analysis.
- `docs/MODEL_ARCHITECTURE.md` — shallow vs. sequence/semantic model analysis.
- `docs/CONTAINER_TRAINER.md` — the A100 ModernBERT ensemble and its fixes.
- `docs/LLM_EXTRACTOR.md` — the (negative) LLM-as-extractor investigation, in full.
- `docs/SOLUTION.md` — detailed methodology, features, and data schema.
- `literature/` — a catalog of ~33 ideas synthesized from ~290 tutoring/education papers.
