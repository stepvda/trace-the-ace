"""Train a few DIVERSE classical variants (objective-grouped OOF) for a
shift-robust ensemble. Saves cache/oof_<name>.parquet (response_id, p, y).
Variants differ in model / regularization to add decorrelation.
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


def main():
    ids = pd.read_csv(os.path.join(CACHE, "row_ids.csv")).iloc[:, 0].astype(str).tolist()
    num = np.load(os.path.join(CACHE, "num.npy"))
    tfidf = sparse.load_npz(os.path.join(CACHE, "tfidf.npz"))
    svd = np.load(os.path.join(CACHE, "svd256.npy"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    y = lab.loc[ids, "is_correct"].to_numpy(float)
    groups = f.loc[ids, "learning_objective_id"].astype(str).to_numpy()
    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med)
    num_imp = np.where(np.isnan(num), med, num)

    def cv_oof(fit_predict):
        oof = np.zeros(len(y))
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        for tr, va in sgkf.split(num_imp, y, groups):
            oof[va] = fit_predict(tr, va)
        return oof

    def lr_variant(C):
        def fp(tr, va):
            sc = StandardScaler().fit(num_imp[tr])
            X = sparse.hstack([tfidf, sparse.csr_matrix(sc.transform(num_imp))]).tocsr()
            m = LogisticRegression(C=C, max_iter=1000, solver="liblinear").fit(X[tr], y[tr])
            return m.predict_proba(X[va])[:, 1]
        return fp

    def hgb_variant(svd_dim):
        def fp(tr, va):
            X = np.hstack([num_imp, svd[:, :svd_dim]])
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                    l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, random_state=1).fit(X[tr], y[tr])
            return m.predict_proba(X[va])[:, 1]
        return fp

    variants = {"lr_c1": lr_variant(1.0), "lr_c03": lr_variant(0.3), "hgb256": hgb_variant(256)}
    for name, fp in variants.items():
        oof = cv_oof(fp)
        pd.DataFrame({"response_id": ids, "p": oof, "y": y}).to_parquet(os.path.join(CACHE, f"oof_{name}.parquet"))
        print(f"{name}: ll={log_loss(y,np.clip(oof,1e-6,1-1e-6)):.5f} auc={roc_auc_score(y,oof):.4f}", flush=True)


if __name__ == "__main__":
    main()
