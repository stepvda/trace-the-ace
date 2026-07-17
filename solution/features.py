"""Feature engineering for the "Trace the Ace" tutoring-outcomes challenge.

This module is imported by BOTH the training script and the inference `main.py`
so that features are computed identically at train and inference time. Keep it
dependency-light (pandas + numpy + stdlib only) for maximum runtime portability.

Public entrypoint:
    build_features(features_df, transcripts_dir) -> pandas.DataFrame
        indexed by response_id, with numeric feature columns plus the raw text
        columns  text_all / text_student / text_tutor / text_lo  used downstream
        by TF-IDF vectorizers.
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

# ---- keyword lexicons (lowercased, matched as substrings / word-ish) --------
TUTOR_PRAISE = [
    "well done", "good job", "great job", "great work", "excellent", "perfect",
    "exactly", "that's right", "thats right", "correct", "brilliant", "fantastic",
    "spot on", "nailed it", "good work", "nice work", "amazing", "superb",
    "you got it", "that's correct", "thats correct", "well remembered", "lovely",
]
TUTOR_CORRECTIVE = [
    "not quite", "not right", "incorrect", "that's not", "thats not", "try again",
    "almost", "close", "let's try", "lets try", "have another", "not exactly",
    "remember that", "be careful", "careful", "actually", "let me explain",
    "let's look", "lets look", "not the", "isn't", "no,", "hmm", "think again",
]
STUDENT_UNCERTAIN = [
    "i don't know", "i dont know", "idk", "not sure", "i'm not sure", "im not sure",
    "no idea", "confused", "don't understand", "dont understand", "don't get",
    "dont get", "i guess", "maybe", "i think it", "is it", "um", "erm", "not really",
]
STUDENT_AFFIRM = [
    "i see", "got it", "makes sense", "i understand", "understand", "oh", "ah",
    "okay", "ok", "yes", "yeah", "yep", "right", "of course", "i get it",
]
QUESTION_RE = re.compile(r"\?")
WORD_RE = re.compile(r"[A-Za-z']+")
NUM_RE = re.compile(r"\d")

# --- literature-grounded talk-move lexicons (Accountable Talk / TalkMoves) ---
TUTOR_REVOICE = ["so you", "you're saying", "youre saying", "what i hear", "in other words",
                 "so what you", "you think that", "sounds like you", "so you're", "so you mean"]
TUTOR_PRESS = ["why", "how do you know", "how did you", "how do you", "explain", "what makes you",
               "can you show", "show me", "prove", "justify", "how come", "walk me through",
               "tell me why", "what makes", "how would you", "convince me"]
TUTOR_ELICIT = ["what do you think", "can you tell", "what about", "any ideas", "what would",
                "how about", "what could", "give me an example", "what else", "can you think"]
STUDENT_REASON = ["because", "since", "therefore", "that's why", "thats why", "which means",
                  "the reason", "in order to", "so that", "as a result", "due to", "that means"]
STUDENT_PASSIVE = ["yes", "yeah", "yep", "okay", "ok", "no", "nope", "sure", "mmm", "mhm",
                   "uh huh", "i guess", "yes miss", "no miss", "correct", "right"]
_STOP = set(("the a an of to and is are was in on at it that this i you he she we they for "
             "be do so no yes ok okay if or as but not have has had will would can could my "
             "your me him her them us what how why when who which").split())

# --- v3 (catalog-driven): in-session correctness proxy, feedback levels, telling ---
# tightened: only UNAMBIGUOUS correctness signals (broad words like good/okay/yes are noise)
PROXY_CONFIRM = ["correct", "exactly", "that's right", "thats right", "well done", "perfect",
                 "spot on", "you got it", "that's correct", "thats correct", "absolutely right",
                 "that's it", "thats it", "precisely"]
PROXY_CORRECTION = ["not quite", "not right", "incorrect", "try again", "that's not right",
                    "thats not right", "not exactly", "have another go", "that's wrong",
                    "thats wrong", "let's try again", "lets try again"]
TELLING = ["the answer is", "it's ", "its ", "you get", "so it's", "so its", "equals",
           "the answer's", "you would get", "that gives", "so you have", "which gives"]
FB_PROCESS = ["the strategy", "try thinking", "the reason is", "method", "notice that",
              "break it", "because", "the way to", "the step", "in order to", "think about how"]
FB_SELFREG = ["how did you", "check your work", "are you sure", "how do you know",
              "what would you do next", "does that make sense", "can you check", "how could you"]
FB_PERSON = ["good job", "smart", "good girl", "good boy", "clever", "you're good",
             "proud of you", "superstar", "well done"]
_MATHOP_RE = re.compile(r"[+\-*/=<>]|\bplus\b|\bminus\b|\btimes\b|\bdivide")
_MULTISTEP = ["then", "next", "after", "therefore", " if ", " per ", "first", "second", "finally"]
_HMS_RE = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def _time_to_seconds(series: pd.Series) -> np.ndarray:
    """Parse elapsed timestamps. Handles 'HH:MM:SS' (the training format) and
    falls back to full-datetime parsing. Returns float seconds, NaN if unknown."""
    vals = series.astype(str).tolist()
    out = np.full(len(vals), np.nan, dtype=float)
    need_dt = []
    for i, v in enumerate(vals):
        m = _HMS_RE.match(v)
        if m:
            h, mn, s = m.group(1), m.group(2), m.group(3)
            out[i] = int(h) * 3600 + int(mn) * 60 + float(s)
        elif v and v.lower() not in ("nan", "none", ""):
            need_dt.append(i)
    if need_dt:
        try:
            dt = pd.to_datetime([vals[i] for i in need_dt], errors="coerce")
            base = dt.min()
            for k, i in enumerate(need_dt):
                if pd.notna(dt[k]) and pd.notna(base):
                    out[i] = (dt[k] - base).total_seconds()
        except Exception:
            pass
    return out


def _count_any(text: str, phrases) -> int:
    return sum(text.count(p) for p in phrases)


def _n_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _read_transcript(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return df


def _session_features(df: pd.DataFrame) -> dict:
    """Compute numeric + text features for a single session transcript df."""
    role = df.get("role", pd.Series([""] * len(df))).astype(str).str.strip().str.lower()
    content = df.get("content", pd.Series([""] * len(df))).astype(str)

    is_student = role.eq("student").values
    is_tutor = role.eq("tutor").values

    contents = content.values
    lc = np.array([c.lower() for c in contents], dtype=object)
    wcounts = np.array([_n_words(c) for c in contents])

    n_utt = len(df)
    n_student = int(is_student.sum())
    n_tutor = int(is_tutor.sum())

    stud_words = int(wcounts[is_student].sum()) if n_student else 0
    tut_words = int(wcounts[is_tutor].sum()) if n_tutor else 0
    tot_words = int(wcounts.sum())

    stud_wlens = wcounts[is_student] if n_student else np.array([0])
    tut_wlens = wcounts[is_tutor] if n_tutor else np.array([0])

    # timestamps -> duration & gaps (timestamps are elapsed HH:MM:SS)
    secs_all = _time_to_seconds(df.get("timestamp", pd.Series([None] * len(df))))
    dur = mean_gap = med_gap = max_gap = np.nan
    secs = secs_all[~np.isnan(secs_all)]
    secs = np.sort(secs)
    if len(secs) >= 2:
        dur = float(secs[-1] - secs[0])
        gaps = np.diff(secs)
        gaps = gaps[gaps >= 0]
        if len(gaps):
            mean_gap = float(np.mean(gaps)); med_gap = float(np.median(gaps)); max_gap = float(np.max(gaps))

    # turn switching
    switches = int(np.sum(role.values[1:] != role.values[:-1])) if n_utt > 1 else 0

    # keyword signals
    tutor_lc = " \n ".join(lc[is_tutor]) if n_tutor else ""
    student_lc = " \n ".join(lc[is_student]) if n_student else ""
    praise = _count_any(tutor_lc, TUTOR_PRAISE)
    corrective = _count_any(tutor_lc, TUTOR_CORRECTIVE)
    uncertain = _count_any(student_lc, STUDENT_UNCERTAIN)
    affirm = _count_any(student_lc, STUDENT_AFFIRM)
    tutor_q = tutor_lc.count("?")
    student_q = student_lc.count("?")
    student_nums = len(NUM_RE.findall(student_lc))

    # unique word richness for student
    stud_tokens = WORD_RE.findall(student_lc)
    stud_uniq_ratio = (len(set(stud_tokens)) / len(stud_tokens)) if stud_tokens else 0.0

    # last / first student utterance sizes
    stud_idx = np.where(is_student)[0]
    last_student_words = int(wcounts[stud_idx[-1]]) if len(stud_idx) else 0
    first_student_words = int(wcounts[stud_idx[0]]) if len(stud_idx) else 0
    last_role_student = int(role.values[-1] == "student") if n_utt else 0

    # --- recency / dynamics (the quiz follows the session, so the END matters) ---
    order = np.arange(n_utt)
    q = max(1, n_utt // 4)
    last_q = order >= (n_utt - q)                       # final quarter of utterances
    ls_mask = is_student & last_q
    lt_mask = is_tutor & last_q
    lastq_student_lc = " \n ".join(lc[ls_mask]) if ls_mask.any() else ""
    lastq_tutor_lc = " \n ".join(lc[lt_mask]) if lt_mask.any() else ""
    lastq_student_words = int(wcounts[ls_mask].sum())
    lastq_n_student = int(ls_mask.sum())
    lastq_tutor_praise = _count_any(lastq_tutor_lc, TUTOR_PRAISE)
    lastq_tutor_corrective = _count_any(lastq_tutor_lc, TUTOR_CORRECTIVE)
    lastq_student_uncertain = _count_any(lastq_student_lc, STUDENT_UNCERTAIN)
    lastq_student_affirm = _count_any(lastq_student_lc, STUDENT_AFFIRM)

    # student word trajectory: 2nd half vs 1st half (rising engagement?)
    half = n_utt // 2
    first_half = order < half
    sh1 = float(wcounts[is_student & first_half].sum())
    sh2 = float(wcounts[is_student & ~first_half].sum())
    student_word_trajectory = (sh2 - sh1) / (sh1 + sh2 + 1.0)

    # student response latency: gap from a tutor utterance to the next student utterance
    lats = []
    for i in range(1, n_utt):
        if is_student[i] and is_tutor[i - 1]:
            a, bb = secs_all[i], secs_all[i - 1]
            if not (np.isnan(a) or np.isnan(bb)):
                d = a - bb
                if 0 <= d < 600:
                    lats.append(d)
    mean_student_latency = float(np.mean(lats)) if lats else np.nan
    median_student_latency = float(np.median(lats)) if lats else np.nan

    # longest run of consecutive tutor utterances (explaining/rephrasing w/o student)
    max_tutor_run = 0; cur = 0
    for i in range(n_utt):
        if is_tutor[i]:
            cur += 1; max_tutor_run = max(max_tutor_run, cur)
        else:
            cur = 0

    # fraction of student utterances that are substantive (>10 words)
    frac_long_student = float((stud_wlens > 10).mean()) if n_student else 0.0
    # net pedagogical signals
    praise_minus_corrective = praise - corrective
    affirm_minus_uncertain = affirm - uncertain

    # --- literature-grounded: talk moves, self-explanation, uptake, ICAP ---
    tutor_revoice = _count_any(tutor_lc, TUTOR_REVOICE)      # revoicing / restating
    tutor_press = _count_any(tutor_lc, TUTOR_PRESS)          # pressing for reasoning
    tutor_elicit = _count_any(tutor_lc, TUTOR_ELICIT)        # eliciting
    student_reason = _count_any(student_lc, STUDENT_REASON)  # self-explanation connectives

    def _content_words(s):
        return set(w for w in WORD_RE.findall(s) if len(w) > 2 and w not in _STOP)

    uptakes = []; press_resp_words = []; constructive = 0; passive = 0
    for i in range(n_utt):
        if is_student[i]:
            w = wcounts[i]; lci = lc[i]
            if w <= 3:                                          # short acknowledgment (Passive)
                passive += 1
            elif w >= 6 and (any(p in lci for p in STUDENT_REASON) or NUM_RE.search(lci)):
                constructive += 1                               # substantive + reasoning/work (Constructive)
        if is_tutor[i] and i > 0 and is_student[i - 1]:         # uptake: tutor builds on student words
            sw = _content_words(lc[i - 1]); tw = _content_words(lc[i])
            if sw:
                uni = sw | tw
                uptakes.append(len(sw & tw) / len(uni) if uni else 0.0)
        if is_tutor[i] and i + 1 < n_utt and is_student[i + 1] and any(p in lc[i] for p in TUTOR_PRESS):
            press_resp_words.append(wcounts[i + 1])             # depth of response to pressing
    uptake_mean = float(np.mean(uptakes)) if uptakes else 0.0
    press_response_words = float(np.mean(press_resp_words)) if press_resp_words else 0.0

    def _safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    feats = {
        "n_utt": n_utt,
        "n_student": n_student,
        "n_tutor": n_tutor,
        "frac_student_utt": _safe_div(n_student, n_utt),
        "tot_words": tot_words,
        "stud_words": stud_words,
        "tut_words": tut_words,
        "frac_words_student": _safe_div(stud_words, tot_words),
        "words_ratio_st": _safe_div(stud_words, tut_words if tut_words else 1),
        "mean_words_utt": _safe_div(tot_words, n_utt),
        "mean_words_student": float(np.mean(stud_wlens)),
        "mean_words_tutor": float(np.mean(tut_wlens)),
        "max_words_student": float(np.max(stud_wlens)),
        "max_words_tutor": float(np.max(tut_wlens)),
        "std_words_student": float(np.std(stud_wlens)),
        "median_words_student": float(np.median(stud_wlens)),
        "duration_sec": dur,
        "mean_gap_sec": mean_gap,
        "median_gap_sec": med_gap,
        "max_gap_sec": max_gap,
        "words_per_min": _safe_div(tot_words, (dur / 60.0)) if (isinstance(dur, float) and dur and dur > 0) else np.nan,
        "turn_switches": switches,
        "switch_rate": _safe_div(switches, n_utt),
        "tutor_praise": praise,
        "tutor_corrective": corrective,
        "tutor_praise_rate": _safe_div(praise, n_tutor),
        "tutor_corrective_rate": _safe_div(corrective, n_tutor),
        "student_uncertain": uncertain,
        "student_affirm": affirm,
        "student_uncertain_rate": _safe_div(uncertain, n_student),
        "student_affirm_rate": _safe_div(affirm, n_student),
        "tutor_q": tutor_q,
        "student_q": student_q,
        "tutor_q_rate": _safe_div(tutor_q, n_tutor),
        "student_q_rate": _safe_div(student_q, n_student),
        "student_nums": student_nums,
        "student_uniq_ratio": stud_uniq_ratio,
        "last_student_words": last_student_words,
        "first_student_words": first_student_words,
        "last_role_student": last_role_student,
        # recency / dynamics
        "lastq_student_words": lastq_student_words,
        "lastq_n_student": lastq_n_student,
        "lastq_student_words_per_utt": _safe_div(lastq_student_words, lastq_n_student),
        "lastq_tutor_praise": lastq_tutor_praise,
        "lastq_tutor_corrective": lastq_tutor_corrective,
        "lastq_student_uncertain": lastq_student_uncertain,
        "lastq_student_affirm": lastq_student_affirm,
        "student_word_trajectory": student_word_trajectory,
        "mean_student_latency": mean_student_latency,
        "median_student_latency": median_student_latency,
        "max_tutor_run": max_tutor_run,
        "frac_long_student": frac_long_student,
        "praise_minus_corrective": praise_minus_corrective,
        "affirm_minus_uncertain": affirm_minus_uncertain,
        # literature-grounded talk moves / self-explanation / uptake / ICAP
        "tutor_revoice": tutor_revoice,
        "tutor_press": tutor_press,
        "tutor_press_rate": _safe_div(tutor_press, n_tutor),
        "tutor_elicit": tutor_elicit,
        "student_reason": student_reason,
        "student_reason_rate": _safe_div(student_reason, n_student),
        "uptake_mean": uptake_mean,
        "constructive_frac": _safe_div(constructive, n_student),
        "passive_frac": _safe_div(passive, n_student),
        "press_response_words": press_response_words,
        # (v3 catalog features REVERTED — they hurt objective-grouped generalization;
        #  see docs/EXPERIMENT_LOG.md. Kept computed above but not emitted.)
        # text fields for TF-IDF
        "text_student": " ".join(contents[is_student]) if n_student else "",
        "text_tutor": " ".join(contents[is_tutor]) if n_tutor else "",
        "text_all": " ".join(contents),
    }
    return feats


NUMERIC_COLS = None  # filled after first build; kept for reference


def build_features(features_df: pd.DataFrame, transcripts_dir: str) -> pd.DataFrame:
    """Build the per-response feature table.

    features_df must have columns: response_id, session_id, learning_objective
    (learning_objective_id optional). transcripts_dir holds {session_id}.csv.
    """
    features_df = features_df.copy()
    sessions = features_df["session_id"].astype(str).unique()

    # compute session features once per session (responses share a transcript)
    sess_feats = {}
    for sid in sessions:
        path = os.path.join(transcripts_dir, f"{sid}.csv")
        df = _read_transcript(path) if os.path.exists(path) else None
        if df is None:
            sess_feats[sid] = None
        else:
            sess_feats[sid] = _session_features(df)

    rows = []
    empty_text = {"text_student": "", "text_tutor": "", "text_all": ""}
    for _, r in features_df.iterrows():
        sid = str(r["session_id"])
        f = sess_feats.get(sid)
        if f is None:
            f = {"n_utt": 0, "n_student": 0, "n_tutor": 0, **empty_text}
        row = dict(f)
        row["response_id"] = r["response_id"]
        row["learning_objective_id"] = str(r.get("learning_objective_id", ""))
        row["text_lo"] = _safe_str(r.get("learning_objective", ""))
        rows.append(row)

    out = pd.DataFrame(rows).set_index("response_id")
    # ensure numeric columns are float and NaNs are handled by downstream imputers
    text_cols = {"text_student", "text_tutor", "text_all", "text_lo", "learning_objective_id"}
    for c in out.columns:
        if c not in text_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def numeric_feature_columns(df: pd.DataFrame) -> list:
    text_cols = {"text_student", "text_tutor", "text_all", "text_lo",
                 "learning_objective_id", "session_id"}
    return [c for c in df.columns if c not in text_cols]
