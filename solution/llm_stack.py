"""Meta-stacking of the LLM verdict onto the classical base signal.

The locally-validated use of the verdict is "does it add signal on top of what the
model already knows". In the container the richest base we have per train response
is the classical out-of-fold probability (classical_oof.parquet). So we fit a small
gradient-boosted meta-model on [base_prob, verdict_one-hot] -> y over train, and
APPLY it to test only if it beats the base alone on an objective-grouped held-out
split. This is self-gating: a redundant or unhelpful verdict leaves the submission
unchanged (weight/decision falls back to base).

Pure sklearn/numpy so it runs anywhere. Used by main_container.py; testable locally
against the 572-row verdict sample + classical_oof.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _onehot(v):
    v = np.asarray(v, float)
    oh = np.zeros((len(v), 3))
    ok = ~np.isnan(v)
    oh[np.where(ok)[0], v[ok].astype(int)] = 1.0
    return oh


def _meta(max_iter=250):
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06,
            max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=60,
            early_stopping=True, random_state=0)


def evaluate_and_apply(base_train, verdict_train, y_train, groups_train,
                       base_test, verdict_test, log=print, min_gain=5e-4, n_seeds=10):
    """Decide (objective-grouped) whether the verdict improves the base, and if so
    return refined TEST probabilities; else return base_test unchanged.

    base_*   : classical probability per response (train OOF / test)
    verdict_*: ordinal verdict per response (nan allowed)
    Returns (test_probs, info_dict).

    Gating is FAIR: the base+verdict meta-model is compared against a base-ONLY
    meta-model trained with identical CV, so the measured gain isolates the verdict's
    incremental value and is NOT inflated by the meta-model merely recalibrating the
    base (a pure-noise verdict then scores ~0 gain and is correctly rejected).
    """
    base_train = np.asarray(base_train, float)
    y = np.asarray(y_train, float)
    g = np.asarray(groups_train)
    Xb = base_train.reshape(-1, 1)                       # base-only control
    Xv = np.hstack([Xb, _onehot(verdict_train)])         # base + verdict

    # Gate on MULTI-SEED robustness, not a single split. On small near-noise data a
    # single-seed objective-grouped OOF gain has std ~0.002, so a lucky seed can show
    # +0.003 for a signal that averages to zero. We require the MEAN gain to clear the
    # threshold AND a majority of seeds to agree in sign, so a lucky seed can't ship.
    gains = []
    for seed in range(n_seeds):
        oof_b, oof_v = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xv, y, g):
            oof_b[va] = _meta().fit(Xb[tr], y[tr]).predict_proba(Xb[va])[:, 1]
            oof_v[va] = _meta().fit(Xv[tr], y[tr]).predict_proba(Xv[va])[:, 1]
        gains.append(_ll(y, oof_b) - _ll(y, oof_v))
    gains = np.array(gains)
    mean_gain, frac_pos = float(gains.mean()), float((gains > min_gain).mean())
    info = {"mean_gain": mean_gain, "std_gain": float(gains.std()),
            "frac_seeds_positive": frac_pos, "n_seeds": n_seeds,
            "note": "obj-grouped OOF, base-only control, multi-seed"}
    log(f"LLM stack: mean gain={mean_gain:+.5f} +/-{gains.std():.5f} "
        f"({frac_pos:.0%} of {n_seeds} seeds > {min_gain})")

    if not (mean_gain > min_gain and frac_pos >= 0.7):
        log("LLM stack: verdict adds no ROBUST held-out signal -> keep classical unchanged")
        info["applied"] = False
        return np.asarray(base_test, float), info

    # gain confirmed -> fit base+verdict on all train, apply to test
    full = _meta().fit(Xv, y)
    Xte = np.hstack([np.asarray(base_test, float).reshape(-1, 1), _onehot(verdict_test)])
    refined = full.predict_proba(Xte)[:, 1]
    info["applied"] = True
    log(f"LLM stack: applied meta-model to test ({len(refined)} rows)")
    return refined, info
