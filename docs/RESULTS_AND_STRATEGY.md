# Results, Learnings & Strategy

New reader? See **[REVIEW_GUIDE.md](REVIEW_GUIDE.md)** first. This doc is the detailed
leaderboard journey and the calibration analysis.

## Leaderboard journey (Trace the Ace, ~429 participants)

| Job | Model / change | shrink | Public Log Loss | Rank | Note |
|---|---|---|---|---|---|
| id-1002 | TF-IDF + **LO target-encoding** (leaky) + numeric | a=1.0 | 0.6224 | #79 | below where we ended up |
| **id-1019** | enriched feats, **no LO-enc**, keep objective-TF-IDF | a=0.4 | **0.6144** | **#27** | first real model |
| id-1022 | same model, more shrinkage | a=0.12 | 0.6200 | #27 | over-shrunk (see below) |
| id-1570 | same model, **no** shrinkage | a=1.0 | 0.6151 | — | slightly worse than a=0.4 |
| **id-1575** | same model, **a=0.68 + recenter to 0.685** | — | **0.6091** | **#18** | pure-calibration win, +20 places |
| id-1579 | container (ModernBERT ensemble); transformer fell back | — | 0.6087 | #18 | prior-fix (0.7025→0.7136) gain; DL unrealized |

**Net: #79 → #27 → #18. Best public log loss 0.6087.** #1 = 0.6013; AUROC ceiling ≈ 0.63.
The field is near-noise — it spans only ~0.02 log loss.

## What moved #79 → #27 (features + de-leaking)
1. **Found the CV→leaderboard leak.** The first model's session-grouped CV looked great
   (0.534) but the **learning-objective target-encoding was leakage**: it adds a huge boost
   when objectives are shared across folds and **nothing** when objectives are unseen
   (objective-grouped CV: numeric+enc 0.6019 ≈ numeric-only 0.5988). The public test has
   effectively unseen objectives → that "signal" vanished and the model scored *below*
   where it should (0.6224). Removing it: 0.6224 → 0.6144.
2. **Switched validation to objective-grouped CV** (hold out learning objectives) — the
   realistic, leakage-free estimate.
3. **Enriched behavioral features** (talk-moves, recency/last-quarter dynamics, latency,
   trajectory) + objective-text TF-IDF raised objective-grouped **AUC 0.631 → 0.648**.

## What moved #27 → #18: the calibration reframe (this was the big correction)

For a long time this project believed it was scoring *below* a "constant baseline of
0.6088" and therefore had negative transferable signal. **That was a mistake**, and
correcting it produced the biggest single gain.

- **0.6088 was never observed.** It is the entropy of the *train* base rate (0.70),
  computed offline, and it assumes the *test* rate is also 0.70.
- **The three same-model shrink anchors are convex in `a`** (log loss is convex in the
  prediction, which is affine in `a`):

  ```
  shrink_a:   0.12    0.40    1.00
  LogLoss:  0.6200  0.6144  0.6151        (convex; minimum near a≈0.68)
  ```

  Extrapolating to `a=0` (a pure-0.7025 constant) gives **≈ 0.6236**, not 0.6088 — and it
  implies the **test base rate is ≈ 0.685**. (This also explains the earlier paradox that
  a=0.12 scored *worse* than a=0.40: we were shrinking hard toward the *wrong* prior.)
- **Implication:** the model **beats the true constant by ~0.014** and has genuine
  discrimination (AUROC 0.604). The fix — recenter predictions onto 0.685 and use the
  convex-optimum shrink — `p_final = 0.685 + 0.68·(p_raw − 0.7025)` — took the score from
  0.6144 to **0.6091 (#27 → #18) with no new features**, and the fact that it *beat* the
  a=0.68-only prediction (0.6119) confirmed the 0.685 rate on the real test.

**Lessons:** (1) never trust an *offline-computed* baseline as if it were observed;
(2) never assume the shift direction — measure it; (3) log loss is convex in shrink, so 3
anchors pin the optimum.

## Where the remaining gap to #1 lives — DISCRIMINATION, not calibration

An information-budget view: the best constant (at the 0.685 rate) is ≈ 0.6229;
#1 = 0.6013 extracts ~0.022 nats from the transcript; this model extracts ~0.014 (≈65%).
Calibration is now largely recovered; **the remaining ~0.008 gap is a better transcript
representation we don't have.** The leaderboard proves such a representation exists and is
findable — **9 of ~50 teams have AUROC ≥ 0.62, two on a single submission** — almost
certainly a fine-tuned transformer. Most of those teams are *badly calibrated*; combining
their AUROC with this project's calibration would land at ~0.604–0.606 (top 5). **That is
the lever**: get a transformer to actually train and transfer in the container (see
[CONTAINER_TRAINER.md](CONTAINER_TRAINER.md)).

## Deep-learning status
The local machine is an **8 GB M1 (no CUDA)** — it cannot fine-tune real transformers; a
small ELECTRA on a truncated window reached only AUROC 0.58 locally. The only GPU is the
**A100 in the offline container**. A ModernBERT ensemble is built and fallback-safe, but
its one real run **fell back to classical** (the transformer errored — likely a
`transformers`/ModernBERT runtime version issue that never trained on the real hardware).
Its accuracy has therefore never been measured. This is unfinished, not disproven.

## Strategy (3 scored subs/week, best-kept)
1. **Debug + ship the ModernBERT container** — read the id-1579 container log to see why
   the DL leg fell back; fix the runtime issue; re-run. This is the only path to top 5.
2. **Objective-conditional features** (see REVIEW_GUIDE §6) — a genuine, untested blind
   spot (multi-objective sessions, within-session AUROC ≈ 0.49).
3. **Small calibration polish** — the container already switches the recenter pivot to the
   model's own OOF mean (0.7136); further per-segment calibration is bounded small.
4. Changing calibration needs **no retraining** — it is a post-hoc scalar in
   `model.py:predict_pipeline` / `submission/main_container.py`.

**Do NOT** pursue: objective-difficulty features, external Eedi data (license), any
transductive/test-time trick (rules forbid cross-test-sample fitting), or more hand-built
behavioral features (saturated). See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
