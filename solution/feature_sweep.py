"""Close the literature: multi-seed-gated A/B of the UNTESTED catalog features.

The 64 shipped features are static counts/rates + a last-quarter block. The catalog's
remaining untested ideas are all TEMPORAL/DYNAMICS signals the static set can't see:
struggle streaks, confusion resolution, self-correction, guess bursts, affect
trajectory, move-transition entropy, multi-turn coherence, praise composition,
post-error feedback quality. This computes each from the raw transcript and asks:
does it add ROBUST incremental signal over the 64-feature base?

Gate = the hardened one from llm_stack: mean gain over 10 objective-grouped CV seeds
AND >=70% of seeds agree. A lucky single-seed +0.003 (std ~0.002 here) cannot pass.
"""
import os, sys, re, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from features import (numeric_feature_columns, STUDENT_UNCERTAIN, STUDENT_AFFIRM,
                      STUDENT_REASON, TUTOR_PRAISE, TUTOR_CORRECTIVE, TUTOR_PRESS,
                      TUTOR_ELICIT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TRANS = os.path.join(ROOT, "data", "train_transcripts")
CAND_CSV = os.path.join(CACHE, "sweep_candidates.parquet")

SELF_CORR = re.compile(r"\b(wait|no,|actually|i mean|oops|sorry|my bad|let me|hold on|scratch that)\b", re.I)
PERSON_PRAISE = re.compile(r"\b(smart|clever|genius|brilliant|good (girl|boy)|talented|bright)\b", re.I)
_word = re.compile(r"[a-z']+")


def _has(lex, text):
    t = " " + text.lower() + " "
    return any(w in t for w in lex)


def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log(c / tot + 1e-12) for c in counts if c > 0)


def candidates_for_session(sid):
    """Compute the untested dynamics features for one session's transcript."""
    p = os.path.join(TRANS, sid + ".csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d = d[d.role.isin(["student", "tutor"])].reset_index(drop=True)
    if len(d) == 0:
        return None
    roles = d.role.tolist()
    txt = d.content.tolist()
    n = len(d)

    # per-turn student flags
    s_idx = [i for i in range(n) if roles[i] == "student"]
    unc = {i: _has(STUDENT_UNCERTAIN, txt[i]) for i in s_idx}
    aff = {i: _has(STUDENT_AFFIRM, txt[i]) for i in s_idx}

    # 6. unresolved_struggle_streak: longest run of consecutive uncertain student turns
    streak = mx = 0
    for i in s_idx:
        if unc[i] and not aff[i]:
            streak += 1; mx = max(mx, streak)
        else:
            streak = 0
    struggle_streak = mx

    # 14. confusion_resolution: uncertain early -> affirm late (monitoring shift)
    if s_idx:
        half = len(s_idx) // 2 or 1
        early, late = s_idx[:half], s_idx[half:]
        unc_early = np.mean([unc[i] for i in early]) if early else 0.0
        aff_late = np.mean([aff[i] for i in late]) if late else 0.0
        confusion_resolution = aff_late - unc_early
    else:
        confusion_resolution = 0.0

    # 15. self_correction markers (student)
    self_corr = sum(1 for i in s_idx if SELF_CORR.search(txt[i])) / (len(s_idx) + 1)

    # 17. rapid_guess_bursts: consecutive short (<=3 word) student turns
    burst = bmax = 0
    for i in s_idx:
        if len(_word.findall(txt[i].lower())) <= 3:
            burst += 1; bmax = max(bmax, burst)
        else:
            burst = 0
    guess_burst = bmax

    # 19. affect_valence_trajectory: (affirm-uncertain) last third - first third
    if len(s_idx) >= 3:
        k = len(s_idx) // 3 or 1
        f3, l3 = s_idx[:k], s_idx[-k:]
        val = lambda g: np.mean([aff[i] - unc[i] for i in g]) if g else 0.0
        affect_traj = val(l3) - val(f3)
    else:
        affect_traj = 0.0

    # 20. move_transition_entropy: entropy over (role[i],role[i+1]) transitions
    trans = {}
    for i in range(n - 1):
        key = roles[i][0] + roles[i + 1][0]
        trans[key] = trans.get(key, 0) + 1
    move_entropy = _entropy(list(trans.values()))

    # 25. multi_turn_coherence: mean Jaccard of content words between consecutive student turns
    def cw(t):
        return set(w for w in _word.findall(t.lower()) if len(w) > 3)
    js = []
    for a, b in zip(s_idx, s_idx[1:]):
        A, B = cw(txt[a]), cw(txt[b])
        if A or B:
            js.append(len(A & B) / (len(A | B) + 1e-9))
    coherence = float(np.mean(js)) if js else 0.0

    # 9. person_vs_task_praise: person-praise fraction of all tutor praise
    t_idx = [i for i in range(n) if roles[i] == "tutor"]
    praise = [i for i in t_idx if _has(TUTOR_PRAISE, txt[i])]
    person = sum(1 for i in praise if PERSON_PRAISE.search(txt[i]))
    person_praise_frac = person / (len(praise) + 1)

    # 8. post_error_feedback_quality: after a corrective tutor turn, does the NEXT
    # tutor turn press/elicit (elaborate) vs just move on?
    good = tot = 0
    for j, i in enumerate(t_idx):
        if _has(TUTOR_CORRECTIVE, txt[i]):
            tot += 1
            nxt = t_idx[j + 1] if j + 1 < len(t_idx) else None
            if nxt is not None and (_has(TUTOR_PRESS, txt[nxt]) or _has(TUTOR_ELICIT, txt[nxt])):
                good += 1
    post_error_quality = good / (tot + 1)

    return dict(struggle_streak=struggle_streak, confusion_resolution=confusion_resolution,
                self_corr=self_corr, guess_burst=guess_burst, affect_traj=affect_traj,
                move_entropy=move_entropy, coherence=coherence,
                person_praise_frac=person_praise_frac, post_error_quality=post_error_quality)


def build():
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    sess = f.drop_duplicates("session_id")[["session_id"]]
    t0 = time.time(); rows = {}
    for k, sid in enumerate(sess.session_id.astype(str)):
        c = candidates_for_session(sid)
        if c is not None:
            rows[sid] = c
        if (k + 1) % 3000 == 0:
            print(f"  {k+1}/{len(sess)} sessions ({time.time()-t0:.0f}s)", flush=True)
    cand = pd.DataFrame.from_dict(rows, orient="index")
    out = f[["response_id", "session_id"]].copy()
    out = out.join(cand, on="session_id").drop(columns="session_id").set_index("response_id")
    out = out.fillna(0.0)
    out.to_parquet(CAND_CSV)
    print(f"built {out.shape} candidate features -> {CAND_CSV}", flush=True)
    return out


def sweep(n_seeds=10, min_gain=5e-4):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    cand = pd.read_parquet(CAND_CSV) if os.path.exists(CAND_CSV) else build()
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    ids = [i for i in X.index.astype(str) if i in set(cand.index.astype(str))]
    X = X.loc[ids]; cand.index = cand.index.astype(str); cand = cand.loc[ids]
    y = lab.loc[ids, "is_correct"].to_numpy(float)
    g = f.loc[ids, "learning_objective_id"].astype(str).to_numpy()
    base = X[numeric_feature_columns(X)].to_numpy(float)
    med = np.nanmedian(base, 0); med = np.where(np.isnan(med), 0, med)
    base = np.where(np.isnan(base), med, base)

    def L(p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    def mk(): return HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05,
             max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
             early_stopping=True, random_state=0)

    # base-only OOF per seed (shared control)
    splits = {s: list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(base, y, g))
              for s in range(n_seeds)}
    base_ll = {}
    for s in range(n_seeds):
        oof = np.zeros(len(y))
        for tr, va in splits[s]:
            oof[va] = mk().fit(base[tr], y[tr]).predict_proba(base[va])[:, 1]
        base_ll[s] = L(oof)
    print(f"base (64 feats) mean OOF logloss over {n_seeds} seeds: "
          f"{np.mean(list(base_ll.values())):.5f}", flush=True)

    cols = list(cand.columns)
    tests = [(c, cand[[c]].to_numpy(float)) for c in cols]
    tests.append(("ALL_9_GROUP", cand.to_numpy(float)))
    print(f"\n{'candidate':<22} {'mean_gain':>10} {'std':>8} {'%seeds+':>8}  verdict", flush=True)
    results = []
    for name, add in tests:
        M = np.hstack([base, add])
        gains = []
        for s in range(n_seeds):
            oof = np.zeros(len(y))
            for tr, va in splits[s]:
                oof[va] = mk().fit(M[tr], y[tr]).predict_proba(M[va])[:, 1]
            gains.append(base_ll[s] - L(oof))
        gains = np.array(gains)
        fp = (gains > min_gain).mean()
        ok = gains.mean() > min_gain and fp >= 0.7
        print(f"{name:<22} {gains.mean():>+10.5f} {gains.std():>8.5f} {fp:>7.0%}  "
              f"{'*** SURVIVES ***' if ok else 'reject'}", flush=True)
        results.append((name, gains.mean(), gains.std(), fp, ok))
    surv = [r for r in results if r[4]]
    print(f"\n=> {len(surv)} of {len(tests)} survive the multi-seed gate: "
          f"{[r[0] for r in surv] or 'NONE'}", flush=True)


if __name__ == "__main__":
    if "build" in sys.argv:
        build()
    else:
        sweep()
