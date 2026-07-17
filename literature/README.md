# Literature — signals of learning in tutoring dialogue

Curated research behind the modeling choices for *Trace the Ace* (predict whether
a student answers the next assessment question correctly from a tutoring
transcript). Each entry links to a **concrete model idea**. PDFs are in
[`papers/`](papers/). All are open-access; classic paywalled works are cited but
not redistributed.

> **Honest framing.** This task is near-noise (field-wide AUROC ≈ 0.63; #1 log
> loss 0.6013 vs. 0.6088 constant baseline). Literature-grounded features mostly
> improve **feature quality and cross-tutor transfer** — they don't break the
> intrinsic-predictability ceiling. Transfer is exactly our measured weakness
> (objective-grouped CV AUC 0.648 → leaderboard 0.604), so theory-grounded,
> tutor-agnostic features are well-aimed.

---

## A. Competition-domain datasets (for pretraining / auxiliary signal)
- **MathDial** — `papers/mathdial_tutoring_dialogue_dataset.pdf` (Macina et al., EMNLP-F 2023; CC-BY-SA). Tutor–student math dialogues with pedagogical annotations. → **Used**: domain-adaptive MLM warmup corpus (`solution/build_pretrain_corpus.py`).
- **ConvoLearn** — `papers/convolearn_constructivist_dialogue_dataset.pdf`. Constructivist tutor–student dialogue. → more pretraining corpus; constructivism annotations align with ICAP features.
- **Eedi / NeurIPS 2020 Diagnostic Questions** — `papers/eedi_neurips2020_diagnostic_questions.pdf` (Wang et al.). **Same provider as part of the test set.** → what predicts correctness on *Eedi* data (question difficulty priors, student-ability signals); informs transfer to the Eedi portion of the shift.
- **FoundationalASSIST** — `papers/foundational_assist_knowledge_tracing.pdf`. KT dataset + LLM grounding.

## B. Talk moves / dialogic teaching → replace ad-hoc keyword features
- **The TalkMoves Dataset** — `papers/talkmoves_dataset_k12_math_discursive_moves.pdf` (Suresh et al., LREC 2022). 567 K-12 math transcripts annotated for **teacher & student discursive moves** (Accountable Talk theory); trained classifiers reach ~73% F1. GitHub: SumnerLab/TalkMoves. → **Top idea**: detect tutor moves (*revoicing, pressing for reasoning, eliciting, restating*) and student moves (*making a claim, providing evidence, relating to another's idea*) instead of my praise/uncertainty keyword lists — more predictive and **tutor-agnostic** (better transfer). A pretrained talk-move classifier could be bundled.
- **Enhancing Talk Moves Analysis in Math Tutoring** — `papers/talk_moves_analysis_math_tutoring.pdf`. Cross-domain (classroom→tutoring) talk-move transfer — directly relevant to our shift.
- **EduCoder** — `papers/educoder_annotation_system.pdf`. Education-transcript annotation schemes (feature taxonomy reference).

## C. Cognitive engagement → the ICAP feature family
- **The ICAP Framework** — `papers/chi_wylie_2014_icap_framework.pdf` (Chi & Wylie, 2014). Interactive > Constructive > Active > Passive engagement predicts learning. → **Idea**: a principled *student constructiveness* score (generating new content / self-explaining vs. passive receipt), replacing my crude "student word share / unique-token ratio."

## D. In-session correctness & its pitfalls → the transformer's job
- **Catching the Correct-Answer Trap** — `papers/correct_answer_trap_tutor_blindspots.pdf`. A correct *final* answer can mask flawed reasoning; tutors (and models) over-trust it. → **Key nuance**: don't just detect "student said the right number" — weigh the *reasoning*. This is precisely what a transformer reading the dialogue can capture and TF-IDF cannot; motivates the DL path and a **self-explanation** feature.
- **VanLehn (2011), *The Relative Effectiveness of Human Tutoring…*** (Educational Psychologist 46(4); paywalled — cited, not redistributed). Interaction *granularity* (step/substep) drives learning gains. → motivates features on **turn-level back-and-forth depth** around the student's reasoning.

## E. Knowledge tracing → within-session learning trajectory
- **Deep Knowledge Tracing** — `papers/deep_knowledge_tracing.pdf` (Piech et al., NeurIPS 2015).
- **Context-Aware Attentive KT (AKT)** — `papers/attentive_knowledge_tracing_akt.pdf` (Ghosh et al., KDD 2020). Monotonic attention over a learner's response history; +up to 6% AUC. → **Idea**: model the *trajectory* of in-session student engagement/correctness (early vs. late), not just aggregates — complements the recency/dynamics features already added.
- **Knowledge Tracing: A Survey** — `papers/knowledge_tracing_survey.pdf`. Landscape + why dialogue-based KT is under-explored (our angle).

## F. LLM-as-tutor evaluation (recent, competition-adjacent)
- **Training LLM Tutors to Improve Student Learning Outcomes** — `papers/training_llm_tutors_learning_outcomes.pdf`. Optimizes tutor turns for *outcomes* — mirrors our target.
- **BEA 2025 Shared Task: Pedagogical Ability Assessment** — `papers/bea2025_pedagogical_ability_shared_task.pdf`; **AI Tutor Evaluation Taxonomy** — `papers/ai_tutor_evaluation_taxonomy.pdf`; **AITutor-EvalKit** — `papers/aitutor_evalkit_capabilities.pdf`. Eight pedagogical dimensions (mistake identification, actionability, guidance…) usable as **auxiliary multi-task labels** for the transformer.

## G. Methods
- **Sentence-BERT** — `papers/sentence_bert_embeddings.pdf` (Reimers & Gurevych, 2019). Basis for the frozen-embedding path.

---

## Actionable shortlist (ranked; would be validated objective-grouped, then July-14)
1. **Talk-move features** (B) — detect tutor/student discursive moves; replace keyword lists. *Highest expected transfer gain.*
2. **In-session correctness + self-explanation** (D) — reasoning-weighted, via the transformer; already the DL path's target. Add explicit "student justifies (because/so/steps)" features to the classical model too.
3. **ICAP constructiveness** (C) — student generative-contribution score.
4. **Uptake** (B) — lexical/semantic overlap between a student turn and the tutor's next turn (tutor builds on the student).
5. **Engagement trajectory** (E) — early-vs-late in-session dynamics (extends current recency features).

## Licensing
MathDial CC-BY-SA-4.0; TalkMoves — see SumnerLab repo; arXiv preprints per author terms. Competition winners must license code MIT and use only openly/commercially-licensed external data — talk-move & MathDial resources qualify; **Eedi Kaggle data does not** (competition license) and is used here only as *published insight*, not as training data.
