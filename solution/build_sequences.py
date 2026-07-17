"""Build per-session PER-TURN feature sequences for the KT-style sequence model.
Saves cache/seq_X.npy (n_sessions, MAXLEN, F), cache/seq_mask.npy, cache/seq_sids.csv.
Recency: keep the LAST MAXLEN turns (padded at the front).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from features import (_time_to_seconds, TUTOR_PRAISE, TUTOR_CORRECTIVE, STUDENT_UNCERTAIN,
                      STUDENT_AFFIRM, STUDENT_REASON, PROXY_CONFIRM, PROXY_CORRECTION, WORD_RE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MAXLEN = 128
F = 13


def _any(t, lex): return 1.0 if any(k in t for k in lex) else 0.0


def session_seq(df):
    role = df.get("role", pd.Series([""] * len(df))).astype(str).str.lower().values
    content = df.get("content", pd.Series([""] * len(df))).astype(str).values
    secs = _time_to_seconds(df.get("timestamp", pd.Series([None] * len(df))))
    n = len(df)
    is_stu = role == "student"; is_tut = role == "tutor"
    lc = [c.lower() for c in content]
    wc = np.array([len(WORD_RE.findall(c)) for c in content], float)
    feats = np.zeros((n, F), np.float32)
    for i in range(n):
        feats[i, 0] = is_stu[i]; feats[i, 1] = is_tut[i]
        feats[i, 2] = 1.0 - is_stu[i] - is_tut[i]           # background/other
        feats[i, 3] = np.log1p(wc[i])
        lat = 0.0
        if i > 0 and not np.isnan(secs[i]) and not np.isnan(secs[i - 1]):
            d = secs[i] - secs[i - 1]
            lat = min(max(d, 0), 60) / 60.0
        feats[i, 4] = lat
        feats[i, 5] = 1.0 if "?" in content[i] else 0.0
        if is_tut[i]:
            feats[i, 6] = _any(lc[i], TUTOR_PRAISE); feats[i, 7] = _any(lc[i], TUTOR_CORRECTIVE)
        if is_stu[i]:
            feats[i, 8] = _any(lc[i], STUDENT_UNCERTAIN); feats[i, 9] = _any(lc[i], STUDENT_AFFIRM)
            feats[i, 10] = _any(lc[i], STUDENT_REASON)
            # proxy correctness from next tutor turn
            pl = 0.5
            for j in range(i + 1, min(n, i + 3)):
                if is_tut[j]:
                    if _any(lc[j], PROXY_CONFIRM): pl = 1.0
                    elif _any(lc[j], PROXY_CORRECTION): pl = 0.0
                    break
            feats[i, 11] = pl
        else:
            feats[i, 11] = 0.5
        feats[i, 12] = (i + 1) / n
    return feats[-MAXLEN:]


def main():
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    sids = f["session_id"].astype(str).unique()
    tdir = os.path.join(ROOT, "data", "train_transcripts")
    X = np.zeros((len(sids), MAXLEN, F), np.float32)
    mask = np.zeros((len(sids), MAXLEN), np.float32)
    t0 = time.time()
    for k, sid in enumerate(sids):
        p = os.path.join(tdir, f"{sid}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str, keep_default_na=False)
                s = session_seq(df); L = len(s)
                X[k, MAXLEN - L:] = s; mask[k, MAXLEN - L:] = 1.0
            except Exception:
                pass
        if k % 5000 == 0:
            print(f"  {k}/{len(sids)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(CACHE, "seq_X.npy"), X)
    np.save(os.path.join(CACHE, "seq_mask.npy"), mask)
    pd.Series(sids).to_csv(os.path.join(CACHE, "seq_sids.csv"), index=False)
    print(f"saved seq_X {X.shape} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
