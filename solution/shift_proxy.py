"""Local proxy for the train->test PROVIDER shift.

The leaderboard truth: my model beats a constant by +0.027 on train but LOSES to it
on the shifted test. Objective-grouped CV doesn't capture the provider shift, so I
can't iterate. This builds a shift proxy: cluster train sessions into K "domains" by
transcript STYLE (not topic), then leave-one-domain-out — train on K-1 domains,
predict the held-out domain, measure logloss vs the constant ON THAT DOMAIN. If the
model loses to the constant out-of-domain, the proxy reproduces the LB and I can
optimize against it.

Fast proxy model = HGB on [numeric + SVD(text)], which carries the classical's signal.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def load():
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    from features import numeric_feature_columns
    num = X[numeric_feature_columns(X)].to_numpy(float)
    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med)
    num = np.where(np.isnan(num), med, num)
    svd = np.load(os.path.join(CACHE, "svd256.npy"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[X.index.astype(str), "is_correct"].to_numpy(float)
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    obj = f.loc[X.index.astype(str), "learning_objective_id"].astype(str).to_numpy()
    return X, num, svd, y, obj


def make_domains(num, svd, k=8, seed=0):
    """Cluster by STYLE: standardized numeric features (turn structure, verbosity,
    latency, ratios) capture tutoring style better than topic. Add a few SVD dims."""
    from sklearn.preprocessing import StandardScaler
    style = StandardScaler().fit_transform(num)
    feat = np.hstack([style, StandardScaler().fit_transform(svd[:, :16])])
    dom = KMeans(k, n_init=4, random_state=seed).fit_predict(feat)
    return dom


def hgb():
    return HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, random_state=0)


def leave_one_domain_out(M, y, dom, label=""):
    """Train on all-but-one domain, predict the held-out domain, compare to that
    domain's own constant. Returns mean (model - constant) logloss gain across domains
    (NEGATIVE = model loses to constant out-of-domain, i.e. reproduces the LB)."""
    gains, rows = [], []
    for d in sorted(set(dom)):
        tr, te = dom != d, dom == d
        if te.sum() < 50 or len(set(y[tr])) < 2:
            continue
        clf = hgb().fit(M[tr], y[tr])
        p = clf.predict_proba(M[te])[:, 1]
        c = np.full(te.sum(), y[tr].mean())            # constant = train-domains' mean
        ll_m, ll_c = L(y[te], p), L(y[te], c)
        auc = roc_auc_score(y[te], p) if len(set(y[te])) > 1 else float("nan")
        gains.append(ll_c - ll_m)
        rows.append((d, int(te.sum()), y[te].mean(), ll_m, ll_c, ll_c - ll_m, auc))
    print(f"\n=== leave-one-domain-out {label} (gain>0 => model beats constant) ===", flush=True)
    print(f"{'dom':>3} {'n':>6} {'rate':>6} {'ll_model':>9} {'ll_const':>9} {'gain':>8} {'auc':>6}")
    for d, n, r, lm, lc, g, a in rows:
        print(f"{d:>3} {n:>6} {r:>6.3f} {lm:>9.4f} {lc:>9.4f} {g:>+8.4f} {a:>6.3f}")
    mg = float(np.mean(gains))
    print(f"MEAN gain vs constant (out-of-domain) = {mg:+.4f}  "
          f"[{'model loses => reproduces LB' if mg < 0 else 'model still wins'}]", flush=True)
    return mg


if __name__ == "__main__":
    X, num, svd, y, obj = load()
    print(f"loaded n={len(y)} num={num.shape} svd={svd.shape}")
    dom = make_domains(num, svd, k=8)
    import collections
    print("domain sizes:", dict(collections.Counter(dom)))
    M = np.hstack([num, svd[:, :120]])
    # in-domain baseline (random CV) for reference
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(hgb(), M, y, cv=5, method="predict_proba")[:, 1]
    print(f"\nin-domain 5-fold: ll_model={L(y,oof):.4f} ll_const={L(y,np.full_like(y,y.mean())):.4f} "
          f"gain={L(y,np.full_like(y,y.mean()))-L(y,oof):+.4f} auc={roc_auc_score(y,oof):.4f}")
    leave_one_domain_out(M, y, dom, "STYLE domains")
