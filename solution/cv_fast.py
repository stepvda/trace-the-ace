"""Fast CV: fit unsupervised transformers (TF-IDF, SVD, scaler) ONCE on all
train data, then refit only the supervised parts (LR, HGB, LO target-encoding)
per fold. Reports OOF log loss / AUC.  Config mirrors model.DEFAULT_CONFIG.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from features import numeric_feature_columns
from model import DEFAULT_CONFIG, _mk_tfidf, _lo_target_encode, _apply_lo_enc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load(group_by=None):
    group_by = group_by or os.environ.get("GROUP_BY", "session_id")
    col = "learning_objective_id" if group_by.startswith("obj") else "session_id"
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[X.index, "is_correct"].to_numpy(dtype=float)
    groups = f.loc[X.index, col].astype(str).to_numpy()
    print(f"  [CV grouped by {col}]", flush=True)
    return X, y, groups


def precompute(X, cfg):
    t0 = time.time()
    num_cols = numeric_feature_columns(X)
    num = X[num_cols].to_numpy(dtype=float)
    med = np.nanmedian(num, axis=0); med = np.where(np.isnan(med), 0.0, med)
    num_imp = np.where(np.isnan(num), med, num)
    scaler = StandardScaler().fit(num_imp)
    num_scaled = scaler.transform(num_imp)

    v_all = _mk_tfidf(cfg["tfidf_all"]).fit(X["text_all"].fillna(""))
    Xall = v_all.transform(X["text_all"].fillna(""))
    parts = [Xall]
    if cfg["use_student_tfidf"]:
        v_st = _mk_tfidf(cfg["tfidf_student"]).fit(X["text_student"].fillna(""))
        parts.append(v_st.transform(X["text_student"].fillna("")))
    if cfg["use_lo_tfidf"]:
        v_lo = _mk_tfidf(cfg["tfidf_lo"]).fit(X["text_lo"].fillna(""))
        parts.append(v_lo.transform(X["text_lo"].fillna("")))
    text_sparse = sparse.hstack(parts).tocsr()
    print(f"  tfidf done {text_sparse.shape} in {time.time()-t0:.1f}s", flush=True)

    svd_feat = None
    if cfg["svd_components"]:
        t1 = time.time()
        k = min(cfg["svd_components"], text_sparse.shape[1] - 1)
        svd_feat = TruncatedSVD(n_components=k, random_state=0, n_iter=5).fit_transform(text_sparse)
        print(f"  svd done in {time.time()-t1:.1f}s", flush=True)

    lr_base = sparse.hstack([text_sparse, sparse.csr_matrix(num_scaled)]).tocsr()
    return {"num_raw": num, "num_scaled": num_scaled, "lr_base": lr_base,
            "svd_feat": svd_feat, "obj": X["learning_objective_id"].astype(str).values}


def run_cv(config=None, n_splits=5):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    X, y, groups = load()
    P = precompute(X, cfg)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y)); oof_lr = np.zeros(len(y)); oof_hgb = np.zeros(len(y))
    gm = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    t0 = time.time()
    for i, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        # LO target enc from train fold
        enc = _lo_target_encode(P["obj"][tr], y[tr], cfg["lo_smoothing"], gm)
        lo_col = _apply_lo_enc(P["obj"], enc, gm).reshape(-1, 1)
        # LR
        Xlr = sparse.hstack([P["lr_base"], sparse.csr_matrix(lo_col)]).tocsr() if cfg["use_lo_target_enc"] else P["lr_base"]
        lr = LogisticRegression(C=cfg["lr_C"], max_iter=1000, solver="liblinear").fit(Xlr[tr], y[tr])
        plr = lr.predict_proba(Xlr[va])[:, 1]
        # HGB
        hb = [P["num_raw"]]
        if P["svd_feat"] is not None: hb.append(P["svd_feat"])
        if cfg["use_lo_target_enc"]: hb.append(lo_col)
        Xh = np.hstack(hb)
        hgb = HistGradientBoostingClassifier(random_state=0, **cfg["hgb"]).fit(Xh[tr], y[tr])
        phgb = hgb.predict_proba(Xh[va])[:, 1]
        w = cfg["blend_w_lr"]; p = w * plr + (1 - w) * phgb
        oof[va], oof_lr[va], oof_hgb[va] = p, plr, phgb
        print(f"  fold{i}: lr={log_loss(y[va],np.clip(plr,1e-6,1-1e-6)):.5f} "
              f"hgb={log_loss(y[va],np.clip(phgb,1e-6,1-1e-6)):.5f} "
              f"blend={log_loss(y[va],np.clip(p,1e-6,1-1e-6)):.5f} ({time.time()-t0:.0f}s)", flush=True)
    def L(a): return float(log_loss(y, np.clip(a, 1e-6, 1 - 1e-6)))
    gm = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    res = {"lr_ll": L(oof_lr), "hgb_ll": L(oof_hgb), "blend_ll": L(oof), "auc": float(roc_auc_score(y, oof)),
           "predstd_lr": float(oof_lr.std()), "predstd_hgb": float(oof_hgb.std())}
    bw, bll = 0.5, res["blend_ll"]
    for w in np.linspace(0, 1, 21):
        l = L(w * oof_lr + (1 - w) * oof_hgb)
        if l < bll: bll, bw = l, w
    res["best_w_lr"], res["best_blend_ll"] = float(bw), float(bll)
    pbest = bw * oof_lr + (1 - bw) * oof_hgb
    # shrink toward prior scan (calibration safety under distribution shift)
    ba, bsl = 1.0, bll
    for a in np.linspace(0, 1, 21):
        l = L(a * pbest + (1 - a) * gm)
        if l < bsl: bsl, ba = l, a
    res["best_shrink_a"], res["best_shrink_ll"] = float(ba), float(bsl)
    res["baseline_const"] = float(L(np.full(len(y), gm)))
    return res


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    print("config:", cfg, flush=True)
    res = run_cv(cfg)
    print("RESULT " + json.dumps(res), flush=True)
    print(f"baseline(const)=0.60876 | lr={res['lr_ll']:.5f} hgb={res['hgb_ll']:.5f} "
          f"blend={res['blend_ll']:.5f} | best_w_lr={res['best_w_lr']:.2f} -> {res['best_blend_ll']:.5f} | AUC={res['auc']:.4f}", flush=True)
