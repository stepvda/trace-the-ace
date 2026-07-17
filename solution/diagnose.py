"""Diagnose the CV->leaderboard gap.

Fast (no TF-IDF): evaluate numeric + LO-target-encoding models under two CV
groupings — by session_id (what we used) and by learning_objective_id (pessimistic
proxy for unseen objectives at test time). Also reports OOF prediction spread and
the effect of shrinking predictions toward the base rate.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features import numeric_feature_columns
from model import _lo_target_encode, _apply_lo_enc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
y = lab.loc[X.index, "is_correct"].to_numpy(float)
sess = f.loc[X.index, "session_id"].astype(str).to_numpy()
obj = f.loc[X.index, "learning_objective_id"].astype(str).to_numpy()
num_cols = numeric_feature_columns(X)
num = X[num_cols].to_numpy(float)
med = np.nanmedian(num, axis=0); med = np.where(np.isnan(med), 0, med)
num_imp = np.where(np.isnan(num), med, num)
gm = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
print(f"base rate={gm:.4f}  const logloss={log_loss(y, np.full(len(y), gm)):.5f}")
print(f"n_objectives={len(set(obj))}  n_sessions={len(set(sess))}\n")


def L(a): return float(log_loss(y, np.clip(a, 1e-6, 1 - 1e-6)))


def cv(groups, use_enc, use_text_numeric, model="hgb", C=1.0, smoothing=20.0, tag=""):
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(X, y, groups):
        feats = [num_imp]
        if use_enc:
            enc = _lo_target_encode(obj[tr], y[tr], smoothing, gm)
            col = _apply_lo_enc(obj, enc, gm).reshape(-1, 1)
            feats = [num_imp, col]
        M = np.hstack(feats)
        if model == "hgb":
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                    max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
                    early_stopping=True, random_state=0)
            clf.fit(M[tr], y[tr]); oof[va] = clf.predict_proba(M[va])[:, 1]
        else:
            sc = StandardScaler().fit(M[tr])
            clf = LogisticRegression(C=C, max_iter=1000)
            clf.fit(sc.transform(M[tr]), y[tr]); oof[va] = clf.predict_proba(sc.transform(M[va]))[:, 1]
    ll = L(oof); auc = roc_auc_score(y, oof)
    # shrink-to-prior scan
    best = (1.0, ll)
    for a in np.linspace(0, 1, 11):
        ls = L(a * oof + (1 - a) * gm)
        if ls < best[1]: best = (a, ls)
    print(f"  {tag:38s} ll={ll:.5f} auc={auc:.4f} predstd={oof.std():.3f} "
          f"| best_shrink a={best[0]:.1f}->{best[1]:.5f}")
    return oof


print("=== grouped by SESSION (what we used) ===")
cv(sess, False, False, "hgb", tag="numeric-only HGB")
cv(sess, True, False, "hgb", tag="numeric + LO target-enc HGB")
print("\n=== grouped by OBJECTIVE (pessimistic: unseen objectives) ===")
cv(obj, False, False, "hgb", tag="numeric-only HGB")
cv(obj, True, False, "hgb", tag="numeric + LO target-enc HGB")
cv(obj, True, False, "hgb", smoothing=100, tag="numeric + LO enc (smooth=100) HGB")
