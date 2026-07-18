# LLM-as-Extractor — build, method, and results

The single highest-upside idea from two independent analyses (my own
`MODEL_ARCHITECTURE.md` #1 and DeepSeek's #1): use an instruct LLM to read a
tutoring transcript and emit a **mastery signal** the shallow classical model
cannot compute — specifically whether the student **reasoned to the answer
themselves vs was told/led to it** (the "correct-answer trap"), and whether their
reasoning was confident-but-wrong. Lexical praise/uncertainty features are blind to
this; an LLM is not.

This doc records the full build, the design lessons, and — honestly — whether it
actually beat the classical model on held-out objectives.

## Why it *could* beat the classical (the thesis)
The classical model already has literature-grounded talk-move features (pressing for
reasoning, revoicing, self-explanation, tutor uptake) that partially proxy "who did
the thinking." The LLM's potential edge is reading the *semantics* of the exchange:
a right final answer that the tutor spoon-fed should predict a LOWER next-question
success than a student's independent correct explanation — a distinction invisible to
bag-of-words.

## The pipeline (3 components, all unit-tested)
- **`solution/llm_extract.py`** — local validator. Samples a fixed 572-row,
  26-objective slice (seed 0), extracts a verdict per response via a local LLM
  (Ollama, offline), and scores it two ways: (a) verdict alone vs the label,
  (b) incremental over the 64 numeric features, objective-grouped.
- **`solution/llm_verdict_vllm.py`** — container extractor (vLLM/A100). Imports the
  *exact* prompt + turn-selection from the validator, so the A100's larger model
  applies identical, validated logic. One call per session; fully offline; any
  failure is caught so the submission falls back.
- **`solution/llm_stack.py`** — self-gating meta-model. Folds the verdict onto the
  classical base signal and **applies it to the test set only if it beats a
  base-only control on an objective-grouped held-out split**. This is the same
  decision the container makes.

## Three design lessons (found by direct probing)
1. **Tail turns are useless here.** Sessions end in goodbyes ("have a great week,
   bye"). Feeding the *last* N turns gave the model no mastery signal and it
   collapsed to a constant. Fix: select the **substantive middle window** (drop
   intro/outro + pleasantries) — that's where the math work lives.
2. **Small models won't emit a calibrated probability.** Asked for 0-100, a 3B
   returned the *same* number for every transcript (mode collapse) — it pattern-
   matches the task, not the content. A **discrete 3-way verdict**
   (MASTERED/PARTIAL/CONFUSED) is the small-model-safe output that actually varies.
3. **The gating must use a base-only control.** A meta-model that ingests
   `[base_prob, verdict]` will recalibrate the base and show a spurious ~+0.005
   logloss "gain" even for a **pure-noise** verdict. Comparing against a meta-model
   fit on the base alone (identical CV) isolates the verdict's true incremental
   value. Unit-tested: signal→apply, noise→reject.

## Results (objective-grouped, leakage-free)
The verdict is a *fixed external function* (no fitting on train), so its correlation
with the label on held-out rows **is** its transferable signal — no leakage caveat,
unlike objective-derived features.

| Extractor model | Verdict corr w/ y | `llm_stack` gain over classical | Decision |
|---|---|---|---|
| **llama3.2:3b** (local) | +0.021 | −0.0022 (single seed) | ❌ no signal |
| qwen2.5:7b-instruct (local) | — | — | ⏹️ killed: DeepSeek already answered it definitively |
| **deepseek** (frontier, ceiling test) | **−0.071** | **+0.0006 ± 0.0021** (10 seeds; single-seed 42 gave a misleading +0.0030) | ❌ **no ROBUST signal** |

**Verdict distributions (n=572):** 3B → PARTIAL 322 / CONFUSED 247 / MASTERED 3 (near-
constant, 0.03 gap). DeepSeek → CONFUSED 384 (P=0.628) / PARTIAL 176 (0.574) /
MASTERED 12 (0.417) — genuinely discriminating, but **inverted**: the students it
calls CONFUSED succeed *more* than those it calls MASTERED (corr −0.071).

### The definitive finding
The frontier model IS the ceiling test — DeepSeek is far more capable than any 7-8B
we could bundle, so if its verdict carries no robust signal, none will. It doesn't:
- verdict **anti-correlated** with the outcome (−0.071), and it **hurt** over the
  numeric features (−0.0025 AUC);
- over the classical OOF, a single lucky CV seed showed +0.0030, but **across 10
  seeds the gain is +0.0006 ± 0.0021** (range −0.0029…+0.0036, positive in only 40%)
  — statistically **zero**.

**Why the *zero-shot verdict* failed here.** (1) On the *log-loss metric* the field is
near-noise (it spans ~0.02 log loss), so the bar for a *calibration* gain is tiny —
but the real headroom is in **discrimination** (AUROC), and a single distilled 3-way
verdict is far too coarse to move it. (2) Whatever "who did the reasoning" signal a
*zero-shot* read can surface is **already captured** by the classical model's
literature-grounded talk-move / self-explanation / uncertainty features — exactly the
earlier finding that the v3 *lexical* in-session correctness proxy also added nothing.
(3) The inverted sign is a faint hint of a real "productive-struggle / desirable-
difficulty" effect (students who visibly struggle learn more), but it is far too weak
and noisy to exploit *as a hand-labelled verdict*.

**Scope note — this does NOT mean the transcript is signal-exhausted (correction,
2026-07-18).** The null above is specific to the **zero-shot verdict** framing. A later
experiment — a **supervised fine-tuned encoder** (ModernBERT over the focused
objective-centered representation) — extracts materially *more* discrimination than the
classical model: objective-grouped OOF AUROC ≈ **0.674 vs classical 0.645** (~0.63
LB-equivalent vs classical 0.604). So the transcript plainly holds discrimination the
classical features miss; what fails is asking a *frozen* LLM to name it in one word,
not the premise that extra signal exists. (The transformer story — including the
long-hidden sdpa-NaN bug that made it look like there was no signal — lives in
`EXPERIMENT_LOG.md` / `MODEL_ARCHITECTURE.md`.)

### Methodology lesson this surfaced (now baked into `llm_stack`)
A **single-seed** objective-grouped OOF gain on small near-noise data has std ~0.002,
so a lucky seed shows +0.003 for a signal that averages to zero — my original
+0.0005 single-seed gate would have **wrongly shipped** the DeepSeek verdict.
`llm_stack.evaluate_and_apply` now gates on the **mean over 10 seeds AND ≥70% of
seeds agreeing** (verified: synthetic-signal→apply @100%, noise→reject @0%,
DeepSeek→reject @40%). This is a durable guard for any future stacked feature.

## Decision rule (pre-registered, so the result isn't rationalised after the fact)
- **Any model's verdict beats the classical** (`llm_stack` gain > +0.0005,
  objective-grouped) → wire `llm_verdict_vllm` + `llm_stack` into
  `main_container.py`, bundle that model (or the smallest that reproduces the gain),
  smoke-test, ship for July 14.
- **7B and frontier both null** → the classical already captures whatever transcript
  signal a *zero-shot verdict* can surface (consistent with the earlier v3 lexical
  "in-session correctness proxy," which also hurt). Record as a definitive negative
  **for this framing**; do **not** ship a model-bloating step that adds no value.
  Honest > shippable. (A *supervised* fine-tuned model is out of scope for this rule —
  see the closing note.)

## Container integration (built, gated on a positive result)
Bundling plan if it fires: pre-build `train_prompts.parquet` (session→prompt, small)
so the container computes train verdicts with the *same* bundled model that scores
test (consistency), then `llm_stack` fits the meta on `[classical_oof, verdict]` and
self-gates on an objective-grouped split. Disk on the 8 GB M1 (≈10-12 GB free) rules
out a 7B fp16; a 3B fp16 (~6 GB) or a 7B-AWQ 4-bit (~5 GB) fits after reclaiming the
container zip + Ollama cache. Bundle the *same* model validated, for honest parity.

*(This plan never fired — the result was negative. It is also now superseded on the
architecture side: the solution has moved to **pre-train-and-bundle finished weights,
inference-only container**, so any LLM step would ship as validated bundled weights
rather than be computed inside the container at submission time.)*

## LLM-*extractor* (this doc, dead) vs LLM-*classifier* (a different, still-open idea)
This doc is a **definitive negative for the zero-shot LLM-*extractor*** — a *frozen*
instruct LLM emitting a discrete MASTERED/PARTIAL/CONFUSED verdict that we stack onto
the classical model. That approach adds no robust signal and is not shipped.

It is **not** a verdict on using an LLM at all. A distinct, still-open idea (Fable's
Phase-3 moonshot) is an **LLM-*classifier***: a decoder (e.g. QLoRA-fine-tuned
Qwen2.5-7B) trained **directly on the labels** over the same **focused objective-
centered representation** that the fine-tuned encoder already uses successfully. That
is supervised, not zero-shot — the model *learns* the mapping from transcript to
next-question success rather than being asked to name it — so it is not refuted by the
zero-shot null here. It is gated behind the encoder work and only pursued if a strong
transformer base is confirmed (see `RESULTS_AND_STRATEGY.md` / `EXPERIMENT_LOG.md`).
