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

**Why the thesis failed here.** (1) The task is near-noise — the field sits within
0.008 log loss of the constant baseline, so there is very little extractable signal
to begin with. (2) Whatever "who did the reasoning" signal exists is **already
captured** by the classical model's literature-grounded talk-move / self-explanation
/ uncertainty features — exactly the earlier finding that the v3 *lexical* in-session
correctness proxy also added nothing. (3) The inverted sign is a faint hint of a real
"productive-struggle / desirable-difficulty" effect (students who visibly struggle
learn more), but it is far too weak and noisy to exploit.

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
- **7B and frontier both null** → the classical already captures the extractable
  transcript signal (consistent with the earlier v3 lexical "in-session correctness
  proxy," which also hurt). Record as a definitive negative; do **not** ship a
  model-bloating step that adds no value. Honest > shippable.

## Container integration (built, gated on a positive result)
Bundling plan if it fires: pre-build `train_prompts.parquet` (session→prompt, small)
so the container computes train verdicts with the *same* bundled model that scores
test (consistency), then `llm_stack` fits the meta on `[classical_oof, verdict]` and
self-gates on an objective-grouped split. Disk on the 8 GB M1 (≈10-12 GB free) rules
out a 7B fp16; a 3B fp16 (~6 GB) or a 7B-AWQ 4-bit (~5 GB) fits after reclaiming the
container zip + Ollama cache. Bundle the *same* model validated, for honest parity.
