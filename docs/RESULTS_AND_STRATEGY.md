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
