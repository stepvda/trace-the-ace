"""Generate the classical model's objective-grouped OOF predictions (unshrunk)
for all training responses, and bundle them so the container can compare the
DL model against the classical one on a held-out split (to set ensemble weight).
Saves submission/assets/classical_oof.parquet: response_id, p_classical
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ASSETS = os.path.join(ROOT, "submission", "assets")


def main(w_lr=0.55):
    ids = pd.read_csv(os.path.join(CACHE, "row_ids.csv")).iloc[:, 0].astype(str).tolist()
    num = np.load(os.path.join(CACHE, "num.npy"))
    tfidf = sparse.load_npz(os.path.join(CACHE, "tfidf.npz"))
    svd = np.load(os.path.join(CACHE, "svd256.npy"))[:, :120]
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    y = lab.loc[ids, "is_correct"].to_numpy(float)
    groups = f.loc[ids, "learning_objective_id"].astype(str).to_numpy()

    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med)
    num_imp = np.where(np.isnan(num), med, num)

    oof_lr = np.zeros(len(y)); oof_hgb = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for tr, va in sgkf.split(num_imp, y, groups):
        sc = StandardScaler().fit(num_imp[tr])
        Xlr = sparse.hstack([tfidf, sparse.csr_matrix(sc.transform(num_imp))]).tocsr()
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear").fit(Xlr[tr], y[tr])
        oof_lr[va] = lr.predict_proba(Xlr[va])[:, 1]
        Xh = np.hstack([num_imp, svd])
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, random_state=0).fit(Xh[tr], y[tr])
        oof_hgb[va] = hgb.predict_proba(Xh[va])[:, 1]
    oof = w_lr * oof_lr + (1 - w_lr) * oof_hgb
    print("classical OOF (obj-grouped) logloss=%.5f auc=%.4f" % (
        log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)), roc_auc_score(y, oof)))
    out = os.path.join(ASSETS, "classical_oof.parquet")
    pd.DataFrame({"response_id": ids, "p_classical": oof, "y": y}).to_parquet(out)
    print("saved", out)


if __name__ == "__main__":
    main()
