"""Shared text builder for the transformer, used by BOTH the local precompute of
training texts AND the in-container inference (main.py), so train/test texts are
identical.

Text per response = "Objective: <lo>. Dialogue: <last N words, role-tagged>".
Recency window (last N words) because the assessment follows the session end.
"""
import os
import pandas as pd

DEFAULT_N_WORDS = 1600  # ~2048 tokens after the objective prefix


def _role_tag(r):
    r = (r or "").lower()
    return "S" if r == "student" else ("T" if r == "tutor" else "B")


def build_text_for_session(transcript_df, n_words=DEFAULT_N_WORDS):
    if transcript_df is None or len(transcript_df) == 0:
        return ""
    role = transcript_df.get("role", pd.Series([""] * len(transcript_df))).astype(str)
    content = transcript_df.get("content", pd.Series([""] * len(transcript_df))).astype(str)
    lines = [f"{_role_tag(r)}: {c}" for r, c in zip(role, content)]
    words = " ".join(lines).split()
    return " ".join(words[-n_words:])


def build_texts(features_df, transcripts_dir, n_words=DEFAULT_N_WORDS):
    """Return list of texts aligned to features_df rows."""
    cache = {}
    out = []
    for lo, sid in zip(features_df["learning_objective"], features_df["session_id"].astype(str)):
        if sid not in cache:
            p = os.path.join(transcripts_dir, f"{sid}.csv")
            df = None
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p, dtype=str, keep_default_na=False)
                except Exception:
                    df = None
            cache[sid] = build_text_for_session(df, n_words=n_words)
        lo_s = "" if lo is None else str(lo)
        out.append(f"Objective: {lo_s}. Dialogue: {cache[sid]}")
    return out
