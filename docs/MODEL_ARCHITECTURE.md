# Model-architecture analysis: is a shallow model the right fit for these insights?

*Requested exercise: evaluate whether the current best model type best leverages
the literature insights, and what other AI model types could leverage them more —
to compete with or combine with.*

> **⚠️ Update (reconciled with later results — read this).** New reader? The fast overview
> is [REVIEW_GUIDE.md](REVIEW_GUIDE.md) (current best: log loss 0.6087, #18). This analysis
> predates two important findings; its core thesis holds but two recommendations are superseded:
> 1. **The "LLM-as-extractor" (rec #2 below) was subsequently tested to a *frontier* model
>    (DeepSeek) and produced NO robust signal — it is a dead end.** See
>    [LLM_EXTRACTOR.md](LLM_EXTRACTOR.md). Do not pursue it.
> 2. **The transformer-ensemble thesis is CONFIRMED — now on real hardware, not just the
>    leaderboard.** The leaderboard already implied it (9 of ~50 teams reach AUROC ≥ 0.62 — a
>    findable transformer representation — and it's the only path to top 5). The "runtime issue"
>    that made the container fall back to classical is now diagnosed: ModernBERT under
>    `attn_implementation="sdpa"` emits **NaN logits on padded batches**, so every container run
>    silently trained on NaN and reverted to classical. The transformer leg has therefore
>    contributed **zero** to date; **all LB points are classical-only.** With the fix
>    (**flash-attention**, which unpads) the real ModernBERT-base trains and is **strong**:
>    objective-grouped CV AUROC **0.6737** (focused rep) vs classical OOF **0.6446** — ≈ **0.63
>    LB-equivalent vs classical 0.604**, a large discrimination gain. It is **not yet submitted.**
>    The design also pivoted to **pre-train-and-bundle → inference-only** (weights fine-tuned on a
>    rented GPU, bundled; the container runs inference only). See
>    [CONTAINER_TRAINER.md](CONTAINER_TRAINER.md).
>
> The rest — "shallow model is a bottleneck for sequential/semantic signal; combine, don't
> replace" — remains the right framing.

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
| **Transformer over raw dialogue** (ModernBERT/BERT) | Reads turns *in order* → captures talk-moves-in-context, in-session correctness, reasoning quality *implicitly*; domain-adaptable on tutoring corpora | **Strongly validated**: real ModernBERT-base scores objective-grouped CV AUROC **0.6737** (focused rep) vs classical OOF **0.6446** (≈0.63 LB-equiv vs 0.604); decorrelated from classical | Needs GPU (rented 4090). **Requires flash-attention** — sdpa NaNs on padded batches; 8192-token full-context is rejected (OOMs the budget, drowns signal in mean-pooling) |
| **Knowledge-Tracing sequence model (DKT/AKT)** | *Purpose-built* for "predict next-correct from a sequence of attempts"; models the mastery **trajectory** the shallow model can't | Not yet built; catalog's most-recurring theme | Needs reliable per-turn labels; in-session attempts are sparse/noisy here |
| **Hierarchical utterance→session encoder** | Encode each utterance (move + content) → attention over utterance embeddings → local moves *and* global trajectory; efficient on long transcripts | Not built | More engineering |
| **LLM-as-extractor / judge** | Rates *reasoning quality* & *semantic* in-session correctness — exactly the "Correct-Answer Trap" gap that sank the lexical proxy | **Dead end** as an *extractor*: zero-shot verdicts tested to a frontier model (DeepSeek), no robust signal ([LLM_EXTRACTOR.md](LLM_EXTRACTOR.md)). NB — an LLM-*classifier* (QLoRA decoder fine-tuned on labels over the focused rep) is a **different, still-open** Phase-3 idea | Cost/calibration; unvalidatable locally |
| Graph/relational over dialogue moves | Models references/contingency as edges | Exotic | Low ROI now |

## Compete or combine?
**Combine.** They have complementary error profiles:
- The **shallow classical** is my measured *robustness* strength — it transfers
  under the train→test shift better than fragile deep signals, and it's cheap.
- The **transformer / KT / LLM** capture the *sequential + semantic* structure the
  shallow model cannot.

Two ways to combine (both recommended):
1. **Ensemble** (weighted by held-out performance) — classical + ModernBERT, weight
   chosen on an objective-grouped hold-out, with a classical fallback. The container
   *design* existed but the transformer leg never trained (sdpa-NaN → classical every
   run); it is being rebuilt as **pre-trained, bundled weights in an inference-only
   container** (a 5–6-seed ensemble across the focused + additive-history reps). My data
   shows the transformer is **decorrelated**, so this is a genuine robustness+accuracy gain.
2. **Deep model as a *feature generator* for the shallow model** — e.g., an
   LLM-extracted "understanding score" or a KT "mastery estimate" becomes **one
   semantic feature** in the classical model. This is the fix for why v3 failed:
   replace the **lexical** correctness proxy (which hurt) with a **semantic** one
   from a model that reads meaning. Best of both: robustness of trees + semantics
   of the LLM.

## Recommendation (ranked, for the next window)
1. **Ship the transformer+classical ensemble** — the only validated "better model
   type," now unblocked by the flash-attention fix. Pre-fine-tune ModernBERT-base on a
   rented GPU (5–6 seeds, focused + additive-history reps), OOF-gate it, and bundle the
   finished weights into an **inference-only** container. *Not* "already built" — the prior
   in-container trainer never actually trained the transformer (sdpa-NaN → classical).
2. ~~**LLM-as-extractor**~~ — **dead end** (zero-shot extraction produced no robust signal
   even at frontier scale; see [LLM_EXTRACTOR.md](LLM_EXTRACTOR.md)). The still-open variant
   is an LLM-**classifier** (QLoRA decoder fine-tuned on the labels over the focused rep) — a
   Phase-3 diversity bet, gated on confirming a strong transformer base first.
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
