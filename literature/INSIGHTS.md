# Literature Study — Insights Catalog

Synthesized from **317 ideas** extracted across **290 harvested sources** (16-cluster harvest + 21-agent deep-read + synthesis). Each item is transcript-computable for predicting next-answer correctness. Sources in [`papers/`](papers/) & [`papers2/`](papers2/); raw list in [`harvest_sources.json`](harvest_sources.json); catalog in [`INSIGHTS_catalog.json`](INSIGHTS_catalog.json).

> **Status: this catalog has been mined and is effectively EXHAUSTED as a source of new
> *hand-built features*.** The high-value, feature-mappable ideas were tested: the
> behavioral/talk-move ones (revoicing, pressing-for-reasoning, eliciting, self-explanation)
> **helped** and are shipped; a later batch of "dynamics" features and a lexical
> correctness-proxy **did not transfer** (they land in the noise on this near-noise task).
> The recurring KT/correctness-trajectory theme (theme #1 below) is real but a *shallow*
> model can't exploit it — it needs a sequence/semantic model (see
> [../docs/MODEL_ARCHITECTURE.md](../docs/MODEL_ARCHITECTURE.md)). For a reviewer, the useful
> remaining direction from this literature is **objective-conditional / order-aware modeling
> via the transformer**, not more scalar features. See
> [../docs/REVIEW_GUIDE.md](../docs/REVIEW_GUIDE.md).

## Recurring themes

- Reconstruct a per-attempt correctness proxy from the tutor's confirm-vs-correct reaction to each student answer, then derive recency/streak/two-count/latent-mastery features from that sequence. This is the single most-recurring idea across the corpus (DKT, PFA, BKT, RPFA, Scarlatos KT-in-dialogue all reduce to it) and the closest label-free analogue of the target.
- Anchor features to the SPECIFIC upcoming concept/item, not global session aggregates: concept-recency lag (time/turns since the assessed terms were last touched), practice count on the target KC, and how much the session actually covered the question's content.
- Item difficulty is a required covariate: near-noise correctness is largely mastery x item-difficulty, so surface-complexity of the upcoming question text (readability, symbolic density, rare-word ratio, step count) is a driver none of the current transcript-side features capture.
- Feedback-quality typology beyond praise/corrective keyword presence: Hattie levels (task/process/self-regulation/person), person-vs-task praise direction, post-error elaboration, and mistake pinpoint/actionability.
- Contingency and moderation as INTERACTIONS rather than main-effect counts: tutor move x running student mastery, support-after-error minus support-after-success, early-ability x discourse move, elaboration x participation.
- Sequence/order over bag-of-counts: move-transition bigrams, trailing streaks, first->second-half trajectories, and soft exponential recency kernels replacing hard last-quarter windows.
- Distinguish productive from unproductive struggle: confusion resolution vs persistence, wheel-spinning / unresolved-error streaks, self-correction/self-monitoring, and executive vs instrumental help-seeking.
- Tutor-agnostic normalization for transfer: within-session/within-tutor z-scores, student:tutor ratio forms, and cohort percentile-ranks so verbosity/style baselines cancel.
- Student-side depth-of-processing lexical signals: disfluency, specificity, evidence-grounding, content novelty, initiative, argument completeness (claim+warrant), and challenge-with-justification.
- Answer leakage / tutor 'telling' as a first-order confound on the label itself: revealing the answer inflates next-correct without learning and changes what the target measures.
- Near-noise pipeline discipline: strict prefix-only (leakage-safe) feature construction, session-grouped CV, probability calibration with abstention, coefficient shrinkage sized to the small true effect, and a feature-reliability gate.
- Affect/confusion dynamics are directional, not level: valence trajectory (recovering vs deteriorating), frustration/giving-up markers, and resolved-then-correct episodes.
- Optional heavier semantic layers with high potential value but real effort/reliability caveats: sentence-embedding pooling + semantic uptake, and a frozen LLM role-playing the student to estimate P(next correct) or scoring a discourse rubric / knowledge-state summary.

## Ranked catalog

| # | Feature/method | Value | Effort | Status | How to compute (short) |
|---|---|---|---|---|---|
| 1 | proxy_correctness_history | high | structural | **IMPLEMENTED (v3)** | Walk turns in timestamp order. For each student turn that answers a preceding tutor question, read the immediately-following tutor turn: label the answer 1 if i… |
| 2 | concept_recency_lag | high | structural | candidate | Tokenize the upcoming question stem (or its stated learning-objective text) into lemmatized, stopword-stripped content terms. Scan backward through prior turns … |
| 3 | upcoming_question_content_coverage | medium | structural | **IMPLEMENTED (v3)** | Extract content terms/keyphrases from the upcoming question (lemmatized, stopword-removed). Coverage = fraction appearing anywhere in the transcript; also a stu… |
| 4 | upcoming_question_difficulty_proxy | medium | lexical | **IMPLEMENTED (v3)** | From the next question's text compute: token count; words-per-sentence; mean word length / syllables-per-word (Flesch-Kincaid); symbolic density = (digits + mat… |
| 5 | telling_vs_eliciting_and_answer_leakage | medium | structural | **IMPLEMENTED (v3)** | Classify each tutor turn: telling = declarative/imperative answer or full-procedure statement (the answer is/it'?s/you get/so it equals/= <value>/multiply then.… |
| 6 | unresolved_struggle_streak | medium | structural | **IMPLEMENTED (v3)** | Segment into problem episodes at new-problem cues (tutor turn with next/new problem/let'?s try/okay so, or a topic shift = low content-word overlap with the pri… |
| 7 | feedback_level_composition | medium | lexical | **IMPLEMENTED (v3)** | Classify each tutor turn by lexicon into: task/product (correct/the answer is/that'?s wrong); process/strategy (the strategy/try thinking about/the reason is/me… |
| 8 | post_error_feedback_quality | medium | structural | candidate | For tutor turn(s) immediately after a proxy-incorrect student answer compute: word count (elaboration length); explanation connective present (because/so/which … |
| 9 | person_vs_task_praise_ratio | low | lexical | candidate | Within tutor praise turns, split person/ego-directed (praise token + 2nd-person pronoun / trait adjective, no task referent: you're smart/good boy/you're great)… |
| 10 | tutor_contingency | medium | structural | candidate | Using proxy-correctness labels: (a) support_after_error - support_after_success, where support = tutor-turn length + count of scaffolding/hint markers on the fo… |
| 11 | move_by_mastery_interactions | medium | structural | candidate | Set proxy_mastery = running share of proxy-correct attempts up to each point. Build interaction features: press_density x proxy_mastery, revoice_density x (1-pr… |
| 12 | feedback_delay | medium | structural | candidate | For each student answer turn, measure intervening turns AND elapsed seconds until the next tutor turn carrying evaluative/corrective content. Aggregate mean/med… |
| 13 | student_initiative_and_novelty | medium | structural | candidate | For each student turn label 'initiating' if it opens with a question, introduces on-topic content words absent from the immediately preceding N tutor turns (nov… |
| 14 | confusion_resolution_and_monitoring_shift | low | structural | candidate | Detect confusion onset in student turns (confused/don'?t get/understand/lost/huh/repeated hedges/'?'). Mark RESOLVED if within K turns a student turn is proxy-c… |
| 15 | student_self_correction_markers | low | lexical | candidate | Regex over student turns for self-initiated repair/monitoring: wait/no wait/actually/I mean/let me redo/recheck/try again/I made a mistake/oh I see/that'?s not … |
| 16 | help_seeking_and_formative_initiation | low | lexical | candidate | Executive help cues in student turns (just tell me/what'?s the answer/give me the answer/I give up/can you just do it) vs instrumental (why/how do i/can you exp… |
| 17 | rapid_guess_bursts | low | structural | **IMPLEMENTED (v3)** | Flag a student answer turn glib-fast if response latency (timestamp delta from the prior tutor turn) is below the student's own session median AND the turn has … |
| 18 | student_disfluency_density | low | lexical | candidate | Regex over student turns for filled pauses (\bu[mh]+\b, \ber+\b, hmm), immediate word repetitions (\b(\w+)\s+\1\b), self-repair (I mean/no wait/or rather/--), t… |
| 19 | affect_valence_trajectory | low | lexical | candidate | Score each student turn valence = positive markers (got it/makes sense/oh nice/I see) minus frustration/negative markers (ugh/this is hard/stuck/confusing/give … |
| 20 | move_transition_and_act_entropy | low | lexical | candidate | Tag each turn with a coarse move/act via lexical+punctuation rules (question='?', tell, revoice/restate, praise, correct, backchannel, directive). Emit normaliz… |
| 21 | question_cognitive_level | low | lexical | candidate | Classify each interrogative turn: higher-order/authentic (why/how/explain/justify/what if/compare/predict/what do you think/how could/why might) vs recall/close… |
| 22 | question_chain_and_step_granularity | low | structural | candidate | Find runs of tutor question turns separated only by short student responses within one episode -> mean/max tutor question-chain length and count of chains >=2 e… |
| 23 | student_content_density_and_grounding | low | lexical | candidate | Per student turn: content density = (domain/content tokens: numerals, operation words, learning-objective vocab, contentful nouns/verbs - backchannel/filler ok/… |
| 24 | complete_argument_and_challenge_justify | low | lexical | candidate | Count student turns with claim+warrant co-occurrence (a declarative answer clause AND a justification clause because/so/therefore/which means/that'?s why in the… |
| 25 | multi_turn_coherence | low | structural | candidate | For each turn t compute content-lemma Jaccard between t and the union of turns t-1..t-3 (stopwords removed); average across the session and take its first->seco… |
| 26 | worked_example_and_generation_first | low | structural | candidate | Flag tutor worked-example turns (long, >=2 sequence markers + a result, no preceding student attempt on that content). Flag student independent-attempt turns (m… |
| 27 | interactive_coconstruction | low | structural | candidate | Tag a student turn Interactive if it (a) has high lexical overlap with the immediately-preceding tutor turn (uptake) AND (b) adds new content words not in that … |
| 28 | within_session_and_cohort_normalization | medium | structural | candidate | For every count/rate feature add: its within-session z-score or rank; its student:tutor ratio form (so absolute chattiness cancels); and its cohort percentile-r… |
| 29 | bkt_running_mastery_filter | medium | structural | candidate | Cluster answer contexts into a few 'skills' by TF-IDF of learning-objective terms. Run a fixed-parameter 2-state BKT/HMM (learn/guess/slip ~0.3/0.2/0.1) over ea… |
| 30 | leakage_safe_prefix_and_grouped_cv | medium | structural | candidate | Build every feature for predicting item t from a strict prefix (turns strictly before the assessment) and audit that no feature reads the answer turn or later t… |
| 31 | exp_decay_recency_and_reliability_gate | low | structural | candidate | For any per-turn scalar x_i (uncertainty, affirmation, latency, valence) compute a soft decayed aggregate sum(w_i*x_i)/sum(w_i) with w_i=exp(-lambda*(t_last - t… |
| 32 | sbert_semantic_embeddings | medium | sequence-model | candidate | Embed each turn with a frozen sentence encoder; aggregate the last-k student turns and (separately) tutor turns by element-wise mean/std/min/max pooling into a … |
| 33 | llm_simulated_student_and_rubric | high | llm | candidate | Prompt a frozen instruction-tuned LLM with the role-tagged transcript prefix plus the upcoming question, asking it to role-play the student and emit either an a… |

## Implemented this pass (v3), measured objective-grouped
- In-session **correctness proxy** (#1) + recency/streak/PFA stats — `proxy_last` corr with target = **0.137** (strongest single feature).
- **Feedback levels** (Hattie #7), **telling-vs-eliciting** (#5), **objective difficulty** (#4), **content coverage** (#3), **rapid-guess latency** (#17).

See [../docs/EXPERIMENT_LOG.md](../docs/EXPERIMENT_LOG.md) for the measured helped/hurt of each batch.