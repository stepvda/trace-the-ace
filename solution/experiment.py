"""Fast objective-grouped (and session-grouped) experiments over cached features.

Loads cached numeric / tfidf / svd256 / embeddings and evaluates classifier
combinations with StratifiedGroupKFold. Reports OOF log loss + AUC + the best
shrink-to-prior. Designed for quick overnight iteration.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load(group_by="objective", emb_tag=None):
    ids = pd.read_csv(os.path.join(CACHE, "row_ids.csv")).iloc[:, 0].astype(str).tolist()
    num = np.load(os.path.join(CACHE, "num.npy"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    y = lab.loc[ids, "is_correct"].to_numpy(float)
    col = "learning_objective_id" if group_by.startswith("obj") else "session_id"
    groups = f.loc[ids, col].astype(str).to_numpy()
    data = {"ids": ids, "num": num, "y": y, "groups": groups}
    if os.path.exists(os.path.join(CACHE, "tfidf.npz")):
        data["tfidf"] = sparse.load_npz(os.path.join(CACHE, "tfidf.npz"))
    if os.path.exists(os.path.join(CACHE, "svd256.npy")):
        data["svd"] = np.load(os.path.join(CACHE, "svd256.npy"))
    if emb_tag:
        for view in ("recent", "student"):
            p = os.path.join(CACHE, f"emb_train_{view}_{emb_tag}.npy")
            if os.path.exists(p):
                data[f"emb_{view}"] = np.load(p)
    return data


def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def run(data, use_tfidf=True, use_svd=True, use_emb=False, svd_dim=256,
        emb_dim=None, lr_C=0.3, w_lr=0.6, n_splits=5, hgb_iter=300):
    y, groups, num = data["y"], data["groups"], data["num"]
    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med)
    num_imp = np.where(np.isnan(num), med, num)
    emb_blocks = []
    if use_emb:
        for v in ("recent", "student"):
            if f"emb_{v}" in data:
                e = data[f"emb_{v}"]
                emb_blocks.append(e[:, :emb_dim] if emb_dim else e)
    emb = np.hstack(emb_blocks) if emb_blocks else None

    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    oof_lr = np.zeros(len(y)); oof_hgb = np.zeros(len(y))
    for tr, va in sgkf.split(num_imp, y, groups):
        # LR on sparse tfidf + scaled numeric (+ emb)
        sc = StandardScaler().fit(num_imp[tr])
        dense_parts = [sc.transform(num_imp)]
        if emb is not None:
            dense_parts.append(emb)
        dense = np.hstack(dense_parts)
        blocks = [sparse.csr_matrix(dense)]
        if use_tfidf and "tfidf" in data:
            blocks.insert(0, data["tfidf"])
        Xlr = sparse.hstack(blocks).tocsr()
        lr = LogisticRegression(C=lr_C, max_iter=1000, solver="liblinear")
        lr.fit(Xlr[tr], y[tr]); oof_lr[va] = lr.predict_proba(Xlr[va])[:, 1]
        # HGB on numeric + svd + emb
        hb = [num_imp]
        if use_svd and "svd" in data: hb.append(data["svd"][:, :svd_dim])
        if emb is not None: hb.append(emb)
        Xh = np.hstack(hb)
        hgb = HistGradientBoostingClassifier(max_iter=hgb_iter, learning_rate=0.06,
                max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
                early_stopping=True, random_state=0)
        hgb.fit(Xh[tr], y[tr]); oof_hgb[va] = hgb.predict_proba(Xh[va])[:, 1]
    oof = w_lr * oof_lr + (1 - w_lr) * oof_hgb
    gm = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    # best blend weight
    bw, bll = w_lr, L(y, oof)
    for w in np.linspace(0, 1, 21):
        l = L(y, w * oof_lr + (1 - w) * oof_hgb)
        if l < bll: bll, bw = l, w
    pbest = bw * oof_lr + (1 - bw) * oof_hgb
    ba, bal = 1.0, bll
    for a in np.linspace(0, 1, 21):
        l = L(y, a * pbest + (1 - a) * gm)
        if l < bal: bal, ba = l, a
    return {"lr": L(y, oof_lr), "hgb": L(y, oof_hgb), "blend": L(y, oof),
            "auc": float(roc_auc_score(y, oof)), "best_w": float(bw), "best_blend": float(bll),
            "best_shrink_a": float(ba), "best_shrink_ll": float(bal),
            "predstd": float(oof.std()), "baseline": L(y, np.full(len(y), gm))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="objective")
    ap.add_argument("--emb_tag", default=None)
    ap.add_argument("--use_emb", action="store_true")
    ap.add_argument("--no_tfidf", action="store_true")
    ap.add_argument("--lr_C", type=float, default=0.3)
    a = ap.parse_args()
    data = load(a.group, a.emb_tag)
    r = run(data, use_tfidf=not a.no_tfidf, use_emb=a.use_emb, lr_C=a.lr_C)
    print(f"[group={a.group} emb={a.use_emb} tfidf={not a.no_tfidf}] " + json.dumps(r))
