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
| id-1579 | container (ModernBERT ensemble); transformer silently NaN'd → classical fallback | — | 0.6087 | #18 | 0.6087 = prior-fix (0.7025→0.7136); **DL contributed zero** — sdpa/padding NaN, see DL status |

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
representation the classical model can't produce.** The leaderboard proves such a
representation exists and is findable — **9 of ~50 teams have AUROC ≥ 0.62, two on a single submission** — almost
certainly a fine-tuned transformer. Most of those teams are *badly calibrated*; combining
their AUROC with this project's calibration would land at ~0.604–0.606 (top 5). **That is
the lever — and as of 2026-07-18 it is unlocked.** The container's transformer had never
actually trained (a silent sdpa/padding NaN — see DL status below); fixed with
flash-attention, the real ModernBERT reaches AUROC **0.674** on objective-grouped holdout
(**≈ 0.63 LB-equivalent** vs classical 0.604) — top-tier discrimination. It is **not yet
submitted**; all LB points to date remain classical-only.

## Deep-learning status (corrected 2026-07-18)

**Root cause of the id-1579 fallback — found.** ModernBERT with
`attn_implementation="sdpa"` on **padded** batches emits **NaN logits** (both bf16 *and*
fp32); unpadded / equal-length batches are fine, so it passed every smoke test. ModernBERT
is built for **flash-attention** (which unpads); its SDPA fallback is broken under padding.
The container trainer (`solution/dl_train.py`) tries `sdpa` first → trained on NaN →
**silently fell back to classical every run**. So the "transformer leg" contributed
**zero** for the entire competition — it was never actually run, not disproven. (This is
the long-unexplained id-1579 fallback.)

**Fix = flash-attention** (`attn_implementation="flash_attention_2"`, forced with no sdpa
fallback in `solution/gpu_mbert.py`); verified that real padded batches now train with
finite, decreasing loss. **Once fixed the transformer works and is strong.** Session-1
3-arm A/B on the *real* ModernBERT-base (objective-grouped holdout, 10k subset, 3 epochs,
1 seed):

| arm | representation | AUROC | logloss |
|---|---|---|---|
| **control** | focused objective-centered rep (max_len 3072) | **0.6737** | 0.5690 |
| **history** | + additive History field (3072) | **0.6798** | 0.5729 |
| full | whole transcript highlighted (max_len 8192) | 0.5621 | 0.6408 |
| — | classical OOF baseline (reference) | 0.6446 | 0.5818 |

Objective-grouped CV over-estimates the LB by ~0.04, so `control` ≈ **0.63 LB-equivalent**
vs classical 0.604 — a large, real discrimination gain. **Full-context (8192) is decisively
rejected** (−0.11 AUROC, 3.6× slower): ModernBERT mean-pools, so pooling over ~5k
mostly-irrelevant tokens drowns the signal — the focused objective-centered rep is a
hand-built attention prior the model can't cheaply relearn (retro-validates "selection over
coverage"). `control` and `history` tie within single-seed noise; both are carried for
ensemble representation-diversity.

Note: the shipped encoder is **stock ModernBERT-base** — domain-adaptive pretraining (DAPT)
was only ever applied to a local DistilBERT proxy, never the model that ships; DAPT on the
transcript corpus is now being tested.

**Compute.** The local machine is an **8 GB M1 (no CUDA)** — it cannot fine-tune real
transformers, and can't even run decision-grade A/Bs (the memory-safe DistilBERT config
undertrains to noise ~0.51; the config that discriminates OOMs the machine). Decision-grade
transformer work now runs on a **rented RunPod GPU** (RTX 4090 24 GB, internet-enabled).

## Strategy (3 scored subs/week, best-kept)

The DL leg is now the primary lever, and the container design has **pivoted**. *Old:*
fine-tune ModernBERT *inside* the container at submission time (a no-local-GPU workaround —
and exactly what hid the silent NaN). *New (adopted):* pre-fine-tune on a rented GPU,
**bundle the finished weights, and ship an inference-only container**. This removes the 6h
in-container training ceiling (ensembles/large models become bundle-able) and, crucially,
lets us validate the **exact shipped weights** on local OOF before upload — killing the
silent-failure class. Inference-time padding-NaN guards (bundled flash-attn wheel +
batch-1 sdpa fallback + classical last-resort) live in
[CONTAINER_TRAINER.md](CONTAINER_TRAINER.md).

1. **Ship the fixed transformer (Phase 1, on a 4090, ~$10–17):** flash-attn fix + focused
   rep + DAPT + a 5–6-seed ensemble (control + history), OOF-gated on the full 35k, bundled
   inference-only. **Ship gate (objective-grouped OOF):** blended (transformer+classical)
   AUROC ≥ classical **+0.015** AND calibrated logloss ≤ classical **−0.002**. This is the
   near-certain value and the path to top 5.
2. **Objective-conditional features** (see REVIEW_GUIDE §6) — a genuine, untested blind
   spot (multi-objective sessions, within-session AUROC ≈ 0.49).
3. **Small calibration polish** — the container already switches the recenter pivot to the
   model's own OOF mean (0.7136); further per-segment calibration is bounded small.
4. Changing calibration needs **no retraining** — it is a post-hoc scalar in
   `model.py:predict_pipeline` / `submission/main_container.py`.

**Gated follow-ons** (only if Phase 1 confirms a strong base): **ModernBERT-large** (now
fits the 4090 since the 8192 window is dead) and a **QLoRA decoder-classifier**
(Qwen2.5-7B fine-tuned on labels over the *focused* rep) as a #1-contention diversity play.
Revised odds: P(material LB gain) ~80%, P(top-10) ~55–60%, P(top-3) ~25%.

**Do NOT** pursue: objective-difficulty features, external Eedi data (license), any
transductive/test-time trick (rules forbid cross-test-sample fitting), or more hand-built
behavioral features (saturated). Note: the zero-shot LLM *extractor* remains a dead end —
this QLoRA *classifier* is a different, still-open idea. See
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
