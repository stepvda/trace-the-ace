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


def _relevant_segment(turns, obj_terms, n_words, proxy_tags):
    """The transcript segment most about the objective, ± context turns. The objective's
    main discussion is usually EARLY (measured median position 0.14), so 'last' (take the
    last overlapping mention) often misses it; 'best' takes the contiguous run with the
    highest TOTAL objective-overlap regardless of position."""
    if not obj_terms:
        return ""
    scores = [len(_content_terms(c) & obj_terms) for _, c in turns]
    hit = [i for i, s in enumerate(scores) if s >= MIN_OVERLAP]
    if not hit:
        return ""
    if SEG_MODE == "last":
        end = start = hit[-1]; hs = set(hit)
        while start - 1 in hs:
            start -= 1
    else:  # "best": contiguous hit-run maximizing total overlap
        runs, s, prev = [], hit[0], hit[0]
        for i in hit[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((s, prev)); s = prev = i
        runs.append((s, prev))
        start, end = max(runs, key=lambda r: sum(scores[r[0]:r[1] + 1]))
    a, b = max(0, start - CTX_TURNS), min(len(turns), end + CTX_TURNS + 1)
    return _last_words(_tagged_lines(turns[a:b], proxy_tags), n_words)


def build_text_for_row(turns, lo, n_words=DEFAULT_N_WORDS, centered=True, proxy_tags=True):
    lo_s = "" if lo is None else str(lo)
    lines = _tagged_lines(turns, proxy_tags)
    if not centered:
        return f"Objective: {lo_s}. Dialogue: {_last_words(lines, n_words)}"
    tail = _last_words(lines, RECENT_WORDS)
    rel = _relevant_segment(turns, _content_terms(lo_s), RELEVANT_WORDS, proxy_tags)
    if rel and rel not in tail:          # objective-relevant content found -> two windows
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


def build_texts(features_df, transcripts_dir, n_words=DEFAULT_N_WORDS, centered=True, proxy_tags=True):
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
                                      centered=centered, proxy_tags=proxy_tags))
    return out
