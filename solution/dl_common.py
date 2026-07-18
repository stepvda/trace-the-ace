"""Shared text builder for the transformer, used by BOTH the local precompute of
training texts AND the in-container inference (main_container.py), so train/test texts
are identical.

Objective-CENTERED text (the fix for the transformer's biggest blind spot):
a session can cover several objectives, but the model is asked about ONE. A plain tail
window shows whatever objective was taught LAST, so for ~24% of rows it contains nothing
about the queried objective (measured: in 59%-multi-objective sessions, the relevant
segment is outside the last-1600-word window 41% of the time). We instead build:

    Objective: <lo>.
    Relevant: <the transcript segment most about THIS objective, role-tagged + T+/T- proxy>
    Recent:   <the session tail, role-tagged + T+/T- proxy>

- "Relevant" is found by content-word overlap with the objective terms (identity-free →
  works on UNSEEN objectives; per-sample → competition-rules-legal).
- T+/T- proxy tags mark the tutor's reaction after a student answer (confirm/advance vs
  correct) — the corpus's most-recurring signal (KT / correctness-trajectory), which a
  sequence model can use but a shallow model can't (scalarizing it hurt the GBM).
- Falls back to a pure tail when no segment matches (single-objective / short chats), so
  single-objective behaviour is unchanged.

`centered`/`proxy_tags` flags exist so the objective-centered representation can be A/B'd
against the previous tail representation.
"""
import os
import pandas as pd
from features import TUTOR_PRAISE, TUTOR_CORRECTIVE, STUDENT_AFFIRM, _STOP, WORD_RE

DEFAULT_N_WORDS = 1600      # total budget when NOT centered (legacy tail)
RELEVANT_WORDS = 600        # budget for the objective-relevant window
RECENT_WORDS = 1000         # budget for the session-tail window
CTX_TURNS = 2               # turns of context on each side of the relevant segment
MIN_OVERLAP = 2             # min objective content-word overlap to call a turn "relevant"

# "History" window: earlier on-objective attempts (chronological), prepended under their
# own budget so the model can see the student's TRAJECTORY on the queried skill — not just
# the last (anchor) discussion. 0 = disabled (byte-identical to the recency-only champion).
# Left-truncation degrades this gracefully: oldest history dies first, recency core survives.
HISTORY_WORDS = 0           # total budget for earlier on-objective runs (0 = off)
HIST_RUN_CAP = 60           # per-run word cap so one long early run can't eat the budget
HIST_CTX = 1                # context turns each side of an earlier run (lands the T+/T- tag)


def _role_tag(r):
    r = (r or "").lower()
    return "S" if r == "student" else ("T" if r == "tutor" else "B")


def _content_terms(s):
    return set(w for w in WORD_RE.findall(str(s).lower()) if len(w) > 2 and w not in _STOP)


def _has(lex, text):
    t = " " + text.lower() + " "
    return any(w in t for w in lex)


def _tag_line(role, content, prev_role, proxy_tags):
    rl = (role or "").lower()
    if proxy_tags and rl == "tutor" and prev_role == "student":
        if _has(TUTOR_CORRECTIVE, content):
            return f"T-: {content}"
        if _has(TUTOR_PRAISE, content) or _has(STUDENT_AFFIRM, content):
            return f"T+: {content}"
    return f"{_role_tag(rl)}: {content}"


def _tagged_lines(turns, proxy_tags):
    out, prev = [], ""
    for role, content in turns:
        out.append(_tag_line(role, content, prev, proxy_tags))
        prev = (role or "").lower()
    return out


def _last_words(lines, n_words):
    return " ".join(" ".join(lines).split()[-n_words:])


SEG_MODE = "last"   # "last" = last overlapping mention (validated, matches bundled train_texts);
#                     "best" = densest segment — under A/B evaluation; switch + regen train_texts if it wins


def _obj_scores(turns, obj_terms):
    return [len(_content_terms(c) & obj_terms) for _, c in turns]


def _hit_runs(scores):
    """Contiguous runs [(start, end), ...] of turns whose objective-overlap >= MIN_OVERLAP,
    in chronological order. runs[-1] is the last (recency-anchor) discussion."""
    hit = [i for i, s in enumerate(scores) if s >= MIN_OVERLAP]
    if not hit:
        return []
    runs, s, prev = [], hit[0], hit[0]
    for i in hit[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((s, prev)); s = prev = i
    runs.append((s, prev))
    return runs


def _relevant_segment(turns, obj_terms, n_words, proxy_tags):
    """The transcript segment most about the objective, ± context turns. 'last' takes the
    last overlapping run (recency anchor — validated to beat 'best' by 0.028 AUROC); 'best'
    takes the contiguous run with the highest TOTAL objective-overlap regardless of position."""
    if not obj_terms:
        return ""
    scores = _obj_scores(turns, obj_terms)
    runs = _hit_runs(scores)
    if not runs:
        return ""
    if SEG_MODE == "last":
        start, end = runs[-1]
    else:  # "best": contiguous hit-run maximizing total overlap
        start, end = max(runs, key=lambda r: sum(scores[r[0]:r[1] + 1]))
    a, b = max(0, start - CTX_TURNS), min(len(turns), end + CTX_TURNS + 1)
    return _last_words(_tagged_lines(turns[a:b], proxy_tags), n_words)


def _history_segments(turns, obj_terms, budget, proxy_tags, exclude=()):
    """Earlier on-objective runs (all but the last/anchor run), newest->oldest until the
    word budget is spent, each capped and given ±HIST_CTX context so the tutor T+/T- reaction
    lands; emitted CHRONOLOGICALLY (oldest first) joined by ' … '. This is the student's
    trajectory on the queried skill — the core knowledge-tracing signal the recency-only
    window discards. Skips any run already contained in the anchor/tail (dedupe)."""
    if not obj_terms or budget <= 0:
        return ""
    runs = _hit_runs(_obj_scores(turns, obj_terms))
    if len(runs) < 2:
        return ""                                  # only the anchor run (or none)
    chosen, spent = [], 0                            # (start_index, tagged_text)
    for (s, e) in reversed(runs[:-1]):              # drop the last run (it's the anchor)
        a, b = max(0, s - HIST_CTX), min(len(turns), e + HIST_CTX + 1)
        seg = _last_words(_tagged_lines(turns[a:b], proxy_tags), HIST_RUN_CAP)
        if not seg or any(seg in x for x in exclude if x):
            continue
        w = len(seg.split())
        if spent + w > budget:
            rem = budget - spent
            if rem >= 8:                            # partial fill only if a meaningful chunk fits
                seg = " ".join(seg.split()[-rem:])
                chosen.append((s, seg))
            break
        chosen.append((s, seg)); spent += w
    if not chosen:
        return ""
    chosen.sort(key=lambda x: x[0])                 # chronological (oldest first)
    return " … ".join(seg for _, seg in chosen)


def build_text_for_row(turns, lo, n_words=DEFAULT_N_WORDS, centered=True, proxy_tags=True,
                       full_context=False):
    lo_s = "" if lo is None else str(lo)
    lines = _tagged_lines(turns, proxy_tags)
    if full_context:
        # FULL transcript, chronological, no window truncation (the tokenizer left-truncates
        # at max_len — ModernBERT's 8192 fits ~median sessions whole). On-objective turns are
        # marked in-place with "* " so the model can attend to the whole trajectory while the
        # queried skill's turns stay salient. Subsumes the History/Relevant windows.
        obj_terms = _content_terms(lo_s)
        hits = set(i for i, s in enumerate(_obj_scores(turns, obj_terms)) if s >= MIN_OVERLAP)
        marked = [("* " + ln) if i in hits else ln for i, ln in enumerate(lines)]
        return f"Objective: {lo_s}. Dialogue: " + " ".join(marked)
    if not centered:
        return f"Objective: {lo_s}. Dialogue: {_last_words(lines, n_words)}"
    tail = _last_words(lines, RECENT_WORDS)
    obj_terms = _content_terms(lo_s)
    rel = _relevant_segment(turns, obj_terms, RELEVANT_WORDS, proxy_tags)
    if rel and rel not in tail:          # objective-relevant content found -> windows
        hist = _history_segments(turns, obj_terms, HISTORY_WORDS, proxy_tags, exclude=(rel, tail))
        if hist:                         # earlier attempts on this objective -> trajectory
            return f"Objective: {lo_s}. History: {hist} Relevant: {rel} Recent: {tail}"
        return f"Objective: {lo_s}. Relevant: {rel} Recent: {tail}"
    # no relevant segment (single-objective / short chat): keep the FULL tail (unchanged
    # from the old representation, just with proxy tags), not the shorter Recent window.
    return f"Objective: {lo_s}. Dialogue: {_last_words(lines, n_words)}"


def _parse_turns(df):
    if df is None or len(df) == 0:
        return []
    role = df.get("role", pd.Series([""] * len(df))).astype(str).tolist()
    content = df.get("content", pd.Series([""] * len(df))).astype(str).tolist()
    return list(zip(role, content))


def build_texts(features_df, transcripts_dir, n_words=DEFAULT_N_WORDS, centered=True, proxy_tags=True,
                full_context=False):
    """Return a list of texts aligned to features_df rows. Caches parsed transcripts per
    session; the per-row text still depends on the queried objective (centered mode)."""
    turns_cache = {}
    out = []
    for lo, sid in zip(features_df["learning_objective"], features_df["session_id"].astype(str)):
        if sid not in turns_cache:
            p = os.path.join(transcripts_dir, f"{sid}.csv")
            df = None
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p, dtype=str, keep_default_na=False)
                except Exception:
                    df = None
            turns_cache[sid] = _parse_turns(df)
        out.append(build_text_for_row(turns_cache[sid], lo, n_words=n_words,
                                      centered=centered, proxy_tags=proxy_tags,
                                      full_context=full_context))
    return out
