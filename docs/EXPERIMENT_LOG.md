# Experiment Log — what helped, what hurt

A running, evidence-based record of every measure tried on *Trace the Ace*, with
its **measured effect**. Metrics are **objective-grouped CV** (leakage-free,
hold out learning objectives) unless a **leaderboard (LB)** score is given.
Baselines: **CORRECTION (verified via anchor convexity):** the "0.6088" constant baseline
was the entropy of the TRAIN rate and was NEVER observed on the LB. The true constant-0.7025
score is ≈ **0.6236** (test base rate ≈ 0.685, not 0.70), so our best **0.6087 BEATS the
constant by ~0.015** (LB AUROC 0.604 = genuine discrimination). The a=0.68 + recenter-to-0.685
calibration (`p→0.685+0.68(p−0.7025)`) achieved this (id-1575 0.6091 → id-1579 0.6087, **#18**).
field #1 = 0.6013 (AUROC ~0.63). The earlier "model loses to constant / signal
reverses" framing was WRONG — transfer is *attenuated (~40% eff), not reversed*.

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
| id-1570 | **same model, NO shrink** (July-14 test of "ship unshrunk") | 1.0 | 0.6151 | — | ❌ slightly worse than a=0.4 |
| **id-1575** | same model, **a=0.68 + recenter to 0.685** (`p→0.685+0.68(p−0.7025)`) | 0.68 | **0.6091** | **#18** | ✅ **new best; #27→#18, −0.0053, pure calibration** |
| id-1579 | container ModernBERT ensemble; **transformer emitted NaN logits → silent classical fallback**, prior pivot 0.7025→0.7136 | 0.68 | **0.6087** | #18 | ✅ tiny calibration gain; DL unrealized — root cause now found (sdpa-NaN on padded batches, see below). **Every LB point to date is classical-only.** |

**Calibration curve — now 3 anchors, and my "ship unshrunk" call was WRONG.** a=0.12→0.6200,
**a=0.4→0.6144 (best)**, a=1.0→0.6151. The minimum is **between a≈0.4 and 0.6, NOT at 1.0**.
I extrapolated from two points that "less shrink is always better" and shipped a=1.0 (id-1570);
the leaderboard disproved it — full confidence overshoots slightly. The later a=0.68 **+ recenter
to 0.685** recalibration (id-1575) then improved the kept best to **0.6091 → 0.6087 (#18)**.
Remaining calibration upside is marginal:
the whole a=0.4→1.0 span is only 0.0007, so a tuned a≈0.5 would gain <0.0004 — not worth a
scored submission. Real upside now lies in the **fixed transformer leg** — the in-container
transformer never actually trained (sdpa-NaN, see below), so this lever is still untapped, not
calibration.

---

## ✅ What HELPED
| Measure | Effect | Evidence |
|---|---|---|
| **Objective-grouped CV** (vs session-grouped) | Prevented leakage-driven overfitting | Session-CV said 0.534; reality 0.6144. The switch is *why* I climbed #79→#27. |
| **Drop learning-objective TARGET ENCODING** | Fixed below-baseline LB | It adds huge session-CV signal but **zero** on unseen objectives (numeric+enc 0.6019 ≈ numeric 0.5988 obj-grouped); removing it: LB 0.6224 → 0.6144. |
| **Keep objective-TEXT TF-IDF** | Generalizable topic difficulty | Dropping it (v2) worsened obj-CV 0.585 → 0.606. |
| **Enriched recency/dynamics features** (last-quarter talk, first→second-half trajectory, response latency, tutor-run length) | obj-CV **AUC 0.631 → 0.648** | Full-model objective-grouped CV. |
| **Literature-grounded features** (talk-moves: pressing-for-reasoning / revoicing / eliciting; self-explanation; tutor uptake; ICAP constructive-vs-passive) | **+0.0102 AUC, +0.0023 logloss** | Numeric-only obj-grouped A/B: 0.5909 → 0.6011 AUC. Tutor-agnostic ⇒ expected to transfer better. |
| **Domain-adaptive MLM warmup on tutoring corpus** (MathDial + our transcripts) | **+0.0088 AUC, +0.0037 logloss** | BERT-mini obj-grouped A/B (18k subset): baseline 0.6171 → adapted **0.6259** AUC; logloss 0.6146 → 0.6109. External-data (MathDial) pretraining transfers. *Caveat: this was on a proxy — the **shipped ModernBERT-base is STOCK**; DAPT was only ever applied to the local DistilBERT proxy (`cache/distilbert_adapted`). DAPT on the shipping encoder is being tested in Session 2.* |
| **Fixed the transformer — real ModernBERT-base finally trains (flash-attn)** | control rep **AUROC 0.6737**, logloss 0.5690 vs classical OOF **0.6446** / 0.5818 (obj-grouped, 10k subset, 3 epochs, 1 seed) | ModernBERT + `attn_implementation="sdpa"` emits **NaN logits on padded batches** (bf16 *and* fp32; equal-length batches are fine, so it passed every smoke test) → the container silently fell back to classical **every run for the whole competition** (the long-unexplained id-1579 fallback). `flash_attention_2` fixes it: real padded batches train with finite, decreasing loss. Obj-grouped CV over-estimates LB by ~0.04 ⇒ control ≈ **0.63 LB-equivalent vs classical 0.604** — a large, real discrimination gain. **This is now the primary lever.** |
| **Focused, objective-centered transformer representation** (input built around the on-objective turns, max_len 3072) | **+0.1116 AUROC** over full-context (0.6737 vs 0.5621) | A hand-built attention prior: ModernBERT mean-pools, so feeding it the focused span beats dumping the whole transcript. Retro-validates the classical "selection over coverage" lesson (last>best, focused>full). Settled design principle. |
| **Classical + transformer ENSEMBLE** | **+0.0080 AUC, +0.0049 logloss** | obj-grouped 3597 held-out: classical 0.6318 → blend **0.6398** (w_transformer=0.33); prediction corr 0.65 ⇒ genuinely decorrelated. (Measured in a local proxy A/B; the in-container transformer that was meant to realize it silently NaN'd — see the flash-attn fix above. Being rebuilt inference-only.) |

## ❌ What HURT (or didn't help)
| Measure | Effect | Evidence |
|---|---|---|
| **Learning-objective target encoding** | LB **worse than baseline** | id-1002 = 0.6224 > 0.6088. Classic CV leakage. |
| **Over-shrinkage** (shrink toward prior too much) | LB got worse | id-1022 (a=0.12)=0.6200 vs id-1019 (a=0.4)=0.6144. Convexity ⇒ optimum is a≥0.4; the model is **well-calibrated at higher confidence**. My initial "shift = overconfidence" assumption was **wrong** (that was a *feature* problem, not calibration). |
| **Heavier LR regularization** (C=0.3 vs 1.0) | obj-CV 0.585 → 0.606 | Over-regularization removed signal. |
| **"v2" config** (drop LO-tfidf, add student-tfidf, C=0.3) | obj-CV worse (0.606) | Composite of the two rows above. |
| **Local deep learning** (ELECTRA-small, 384-token recency, on M1) | AUROC **0.58 < 0.648** classical | A tiny truncated transformer can't match full-transcript TF-IDF + engineered features; the M1 (8 GB, no CUDA) can't train a real one. *(The real ModernBERT-base, once trained on a rented GPU, does beat classical — see the flash-attn row above.)* |
| **Full-context transformer input** (whole 8192-token transcript, on-objective turns merely marked; `full_context` mode in `dl_common.py`) | AUROC **0.5621** (−0.1116 vs the focused rep, 0.6737), **3.6× slower**, flat learning curve (0.549→0.562 while the focused rep hit 0.67 in one epoch) → **DECISIVELY REJECTED** | Mean-pooling over ~5k mostly-irrelevant tokens drowns the signal; localizing it in 8192 tokens from ~8k weak labels is a brutal credit-assignment problem. Coverage loses to selection — the focused rep is a prior the model can't cheaply relearn. |
| **v3 catalog features** (in-session correctness *proxy*, Hattie feedback levels, telling, objective difficulty, content coverage) | **−0.008 to −0.016 AUC → REVERTED** | Numeric obj-grouped: base 0.6011 → +refined 0.5929 → +full 0.5854. Two causes: (a) **objective-derived** features (difficulty/coverage) can't transfer to *unseen* objectives (same lesson as LO target-encoding); (b) the correctness-proxy is redundant with existing praise/affirmation features and its noise swamps the one strong sub-signal (`proxy_last` corr 0.137 alone). **Lesson: not every theory-grounded idea transfers — the *behavioral talk-move* features did, this batch did not.** |
| **Semantic objective difficulty** (DeepSeek #6: ridge on MiniLM objective embeddings, leave-one-objective-out) | **−0.042 AUC → rejected** | Meant to fix v3's difficulty the "right" (non-leaky) way, but the ridge's difficulty correlates only **0.047** with actual objective mean-correctness ⇒ **objective *text* does not predict difficulty**. Proves the objective-difficulty signal is memorization/leakage, not a learnable function. |
| **LLM-as-extractor** (DeepSeek #1 / MODEL_ARCHITECTURE #1: instruct LLM reads transcript → 3-way mastery verdict, targeting the "correct-answer trap" lexical feats miss) | **no robust gain → not shipped** | The flagship idea, tested to the ceiling. 3B (llama3.2) = null (verdict corr **+0.021**). **Frontier model (DeepSeek)** genuinely discriminated (CONFUSED 384 / PARTIAL 176 / MASTERED 12) but verdict was **anti-correlated** (−0.071) and its `llm_stack` gain over the classical was **+0.0006 ± 0.0021 over 10 seeds** (a lucky single seed showed +0.0030) ⇒ statistically zero. If the *strongest* model can't beat the classical, no bundleable 7-8B will. Cause: near-noise task + the "who-reasoned" signal is already captured by the talk-move/self-explanation features (same lesson as the v3 lexical proxy). Full write-up: [LLM_EXTRACTOR.md](LLM_EXTRACTOR.md). **Scope note:** this kills the LLM-as-*extractor* (zero-shot verdicts stacked as a feature); a fine-tuned LLM-*classifier* (QLoRA decoder trained on the labels over the *focused* rep) is a DIFFERENT, still-open idea (a gated Phase-3 moonshot). |
| **9 untested "dynamics" catalog features** (struggle streak, confusion-resolution trajectory, self-correction, guess bursts, affect trajectory, move-transition entropy, multi-turn coherence, person-vs-task praise, post-error feedback quality) | **0 of 10 survive → none shipped** | The last untested literature cluster (`solution/feature_sweep.py`), multi-seed gated over the 64-feature base. Every candidate's incremental logloss gain sits in noise: max **+0.00020 ± 0.0003**, the 9-feature group **+0.00021 ± 0.00045** (30% of seeds), none clears the mean>5e-4 **and** ≥70%-seed gate. Confirms — a *third* time — that added behavioral features are redundant with the shipped talk-move set. **This closes the *classical-feature* catalog: all 33 hand-engineered ideas are now shipped, rejected, or proven-redundant.** (This is NOT signal exhaustion — the transformer leg, silently broken all along, is a large *still-open* lever; see the flash-attn row.) |
| **Running two heavy jobs at once on 8 GB** | Both crawl (swap-thrash) | Process lesson: strictly sequential heavy jobs. |

## ⚪ Neutral / built-but-unvalidated
| Measure | Status |
|---|---|
| Probability **clipping** | No gain (blend already conservative). |
| **A100 in-container fine-tuner** (ModernBERT fine-tuned at submission time, ensembled, classical fallback) | **SUPERSEDED — and it never actually trained** (its transformer NaN'd under sdpa → silent classical fallback every run). Design pivots to **pre-train on a rented GPU, bundle the finished weights, container runs INFERENCE only**, validating the EXACT shipped weights on local OOF before upload. In-container fine-tuning is no longer the design. |
| **Additive History field** (transformer rep variant; `HISTORY_WORDS` knob, default 0 = shipped rep unchanged) | history ≈ control (+0.006 AUROC but +0.004 worse logloss; tied within single-seed noise). Under post-hoc calibration the logloss deficit washes out and the AUROC edge survives, so **both are carried** for representation diversity in the ensemble at zero extra cost. |
| **July-14 submission dry-run** (both zips) | ✅ **ready**. Classical smoke test passes end-to-end (unshrunk cfg confirmed: `shrink_a=1.0`, `use_lo_target_enc=False`). **Found + fixed a real packaging bug**: `package.py` double-nested `base_model` (`assets/base_model/base_model/…`) so the container would have **silently fallen back to classical every run** — nullifying the ensemble. Fixed; verified the model now loads offline from `assets/base_model/` in the *actual* built zip, and the container zip runs end-to-end to a valid `submission.csv` (fallback path). *A **second, deeper** cause of the same silent fallback was later found — the sdpa-NaN bug (the transformer trained on NaN and fell back even when the weights loaded). The inference-only rebuild + offline smoke on the built zip eliminates both.* |
| **Frozen sentence embeddings** | Too slow to finish on M1; inconclusive. |
| **Diverse-model ensemble** | Tooling built; assembly pending. |
| **KT-style sequence model** (GRU over per-turn features + SVD globals) | Works (obj-grouped AUC 0.637, decorrelated corr ~0.62) but **weaker than classical and adds ~nothing** (+0.0006 AUC). A small GRU on the 8GB M1 can't beat the tuned classical; the real version is a bigger sequence/transformer trained on a rented GPU. The transformer *idea* is now **confirmed** by the fixed ModernBERT-base result (AUROC 0.6737 > classical 0.6446), not just this small local model. |

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
6. **Gate on multi-seed robustness, never a single split.** On this small,
   near-noise data a single-seed objective-grouped OOF gain has std ~0.002, so a
   lucky seed shows +0.003 for a signal that averages to zero. The LLM-extractor's
   DeepSeek verdict "passed" a single-seed +0.0005 gate (+0.0030 on seed 42) but was
   +0.0006 ± 0.0021 over 10 seeds — zero. `llm_stack` now requires mean-over-10-seeds
   AND ≥70% of seeds to agree before applying any stacked feature.
7. **Test an idea to its ceiling before believing a null.** A weak local 3B nulling
   the LLM-extractor proved only that *a 3B* can't do it; escalating to a frontier
   model (DeepSeek) turned a maybe into a definitive negative. Match the test's
   strength to the claim before shipping *or* discarding.
8. **Smoke-test the REAL input distribution, and gate on the EXACT artifact you
   ship — never on "the code is built."** ModernBERT + `sdpa` emits NaN on *padded*
   batches but is fine on equal-length ones, so a padding-only failure slipped past
   every smoke test and silently disabled the entire transformer leg for the whole
   competition — the container "trained" on NaN and fell back to classical without
   ever erroring. Two durable rules: exercise the actual failure surface (padded
   batches, real lengths), and validate the exact weights you upload on local OOF.
   "Build-complete" is not "verified"; a silent fallback can null your headline lever
   indefinitely.

## Pending (to append as results land)
- **Session 2 (RunPod 4090, in progress):** DAPT on the transcript corpus applied to
  the *shipping* ModernBERT-base (currently STOCK — DAPT had only ever touched the local
  DistilBERT proxy), then a 5-fold OOF for control + history over the full 35k
  (`gpu_oof.py`, folds reproducing the classical OOF exactly) → `oof_transformer.parquet`;
  local `blend_gate.py` then picks the ship variant + blend weight + calibration.
- **Ship gate** (OOF, full 35k, objective-grouped): blended (transformer + classical)
  AUROC ≥ classical **+0.015** AND calibrated logloss ≤ classical **−0.002**. If it passes:
  train the seed ensemble (3 control + 3 history) on all 35k, bundle the finished weights,
  build the **inference-only** container (flash-attn wheel pinned to the image's
  CUDA/torch, plus a coded batch-size-1 sdpa fallback, classical last-resort fallback),
  offline-smoke the built zip (`HF_HUB_OFFLINE=1`), then submit — this would be the
  **first non-classical submission** in the competition.
