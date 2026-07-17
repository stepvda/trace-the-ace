# Experiment Log — what helped, what hurt

A running, evidence-based record of every measure tried on *Trace the Ace*, with
its **measured effect**. Metrics are **objective-grouped CV** (leakage-free,
hold out learning objectives) unless a **leaderboard (LB)** score is given.
Baselines: **CORRECTION (verified via anchor convexity):** the "0.6088" constant baseline
was the entropy of the TRAIN rate and was NEVER observed on the LB. The true constant-0.7025
score is ≈ **0.6236** (test base rate ≈ 0.685, not 0.70), so our best 0.6144 **BEATS the
constant by ~0.009** (LB AUROC 0.604 = genuine discrimination). Optimal shrink a\*≈0.68 →
LB≈0.6126. field #1 = 0.6013 (AUROC ~0.63). The earlier "model loses to constant / signal
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

**Calibration curve — now 3 anchors, and my "ship unshrunk" call was WRONG.** a=0.12→0.6200,
**a=0.4→0.6144 (best)**, a=1.0→0.6151. The minimum is **between a≈0.4 and 0.6, NOT at 1.0**.
I extrapolated from two points that "less shrink is always better" and shipped a=1.0 (id-1570);
the leaderboard disproved it — full confidence overshoots slightly. Net effect on rank: **none**
(the LB keeps the best score, so 0.6144 still stands). Remaining calibration upside is marginal:
the whole a=0.4→1.0 span is only 0.0007, so a tuned a≈0.5 would gain <0.0004 — not worth a
scored submission. Real upside now lies in the **container ensemble**, not calibration.

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
| **Semantic objective difficulty** (DeepSeek #6: ridge on MiniLM objective embeddings, leave-one-objective-out) | **−0.042 AUC → rejected** | Meant to fix v3's difficulty the "right" (non-leaky) way, but the ridge's difficulty correlates only **0.047** with actual objective mean-correctness ⇒ **objective *text* does not predict difficulty**. Proves the objective-difficulty signal is memorization/leakage, not a learnable function. |
| **LLM-as-extractor** (DeepSeek #1 / MODEL_ARCHITECTURE #1: instruct LLM reads transcript → 3-way mastery verdict, targeting the "correct-answer trap" lexical feats miss) | **no robust gain → not shipped** | The flagship idea, tested to the ceiling. 3B (llama3.2) = null (verdict corr **+0.021**). **Frontier model (DeepSeek)** genuinely discriminated (CONFUSED 384 / PARTIAL 176 / MASTERED 12) but verdict was **anti-correlated** (−0.071) and its `llm_stack` gain over the classical was **+0.0006 ± 0.0021 over 10 seeds** (a lucky single seed showed +0.0030) ⇒ statistically zero. If the *strongest* model can't beat the classical, no bundleable 7-8B will. Cause: near-noise task + the "who-reasoned" signal is already captured by the talk-move/self-explanation features (same lesson as the v3 lexical proxy). Full write-up: [LLM_EXTRACTOR.md](LLM_EXTRACTOR.md). |
| **9 untested "dynamics" catalog features** (struggle streak, confusion-resolution trajectory, self-correction, guess bursts, affect trajectory, move-transition entropy, multi-turn coherence, person-vs-task praise, post-error feedback quality) | **0 of 10 survive → none shipped** | The last untested literature cluster (`solution/feature_sweep.py`), multi-seed gated over the 64-feature base. Every candidate's incremental logloss gain sits in noise: max **+0.00020 ± 0.0003**, the 9-feature group **+0.00021 ± 0.00045** (30% of seeds), none clears the mean>5e-4 **and** ≥70%-seed gate. Confirms — a *third* time — that added behavioral features are redundant with the shipped talk-move set. **This closes the catalog: all 33 ideas are now shipped, rejected, or proven-redundant.** |
| **Running two heavy jobs at once on 8 GB** | Both crawl (swap-thrash) | Process lesson: strictly sequential heavy jobs. |

## ⚪ Neutral / built-but-unvalidated
| Measure | Status |
|---|---|
| Probability **clipping** | No gain (blend already conservative). |
| **A100 container-trainer** (ModernBERT fine-tuned in-container, ensembled, classical fallback) | Built + locally validated (code); ModernBERT-on-A100 accuracy unvalidated (smoke stuck on 692 MB upload; fallback protects). |
| **July-14 submission dry-run** (both zips) | ✅ **ready**. Classical smoke test passes end-to-end (unshrunk cfg confirmed: `shrink_a=1.0`, `use_lo_target_enc=False`). **Found + fixed a real packaging bug**: `package.py` double-nested `base_model` (`assets/base_model/base_model/…`) so the container would have **silently fallen back to classical every run** — nullifying the ensemble. Fixed; verified the model now loads offline from `assets/base_model/` in the *actual* built zip, and the container zip runs end-to-end to a valid `submission.csv` (fallback path). |
| **Frozen sentence embeddings** | Too slow to finish on M1; inconclusive. |
| **Diverse-model ensemble** | Tooling built; assembly pending. |
| **KT-style sequence model** (GRU over per-turn features + SVD globals) | Works (obj-grouped AUC 0.637, decorrelated corr ~0.62) but **weaker than classical and adds ~nothing** (+0.0006 AUC). A small GRU on the 8GB M1 can't beat the tuned classical; the real version is a bigger sequence/transformer on the A100. Architecture *idea* validated in principle by the transformer result, not by this small local model. |

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

## Pending (to append as results land)
- Domain-adaptive pretraining A/B verdict (BERT-mini, MathDial+transcripts).
- Full-model objective-grouped re-check with literature features.
- Final robust ensemble (classical-with-lit + best transformer) objective-grouped.
