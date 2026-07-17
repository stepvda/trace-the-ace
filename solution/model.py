"""Model pipeline for Trace the Ace. Pure numpy/pandas/scipy/scikit-learn so it
runs in the offline competition container.

Two entrypoints used by both CV and inference so behaviour is identical:
    fit_pipeline(config, X_df, y)      -> artifacts (picklable dict)
    predict_pipeline(artifacts, X_df)  -> np.ndarray of P(correct)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from features import numeric_feature_columns

DEFAULT_CONFIG = {
    "tfidf_all": {"max_features": 30000, "ngram_range": (1, 2), "min_df": 5, "sublinear_tf": True},
    "tfidf_student": {"max_features": 20000, "ngram_range": (1, 2), "min_df": 5, "sublinear_tf": True},
    "tfidf_lo": {"max_features": 3000, "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True},
    "svd_components": 120,
    "lo_smoothing": 20.0,
    "lr_C": 1.0,
    "hgb": {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 31,
            "l2_regularization": 1.0, "min_samples_leaf": 40, "early_stopping": True,
            "validation_fraction": 0.1, "n_iter_no_change": 25},
    "blend_w_lr": 0.55,  # weight on LR; HGB gets (1 - w)
    "clip": 0.005,       # clip final probs to [clip, 1-clip]
    "use_student_tfidf": False,  # text_all already contains student utterances
    "use_lo_tfidf": True,
    "use_lo_target_enc": True,
}


def _mk_tfidf(cfg):
    # analyzer='char_wb' gives character n-grams — robust to cross-provider spelling/
    # vocabulary variation (a word-level match requires identical tokens; char n-grams
    # match sub-word morphology, which transfers better under a provider shift).
    return TfidfVectorizer(
        max_features=cfg["max_features"], ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"], sublinear_tf=cfg.get("sublinear_tf", True),
        analyzer=cfg.get("analyzer", "word"),
        strip_accents="unicode", lowercase=True,
    )


def _lo_target_encode(obj_ids, y, smoothing, global_mean):
    df = pd.DataFrame({"o": np.asarray(obj_ids), "y": np.asarray(y, dtype=float)})
    agg = df.groupby("o")["y"].agg(["sum", "count"])
    enc = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
    return enc.to_dict()


def _apply_lo_enc(obj_ids, enc_map, global_mean):
    return np.array([enc_map.get(o, global_mean) for o in obj_ids], dtype=float)


def fit_pipeline(config, X_df, y):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    y = np.asarray(y, dtype=float)
    num_cols = numeric_feature_columns(X_df)
    global_mean = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))

    # --- numeric ---
    num = X_df[num_cols].to_numpy(dtype=float)
    medians = np.nanmedian(num, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    num_imp = np.where(np.isnan(num), medians, num)
    scaler = StandardScaler().fit(num_imp)
    num_scaled = scaler.transform(num_imp)

    # --- text vectorizers ---
    # transcript TF-IDF (text_all/text_student) is provider-SPECIFIC vocabulary and
    # is the prime suspect for the train->test transfer failure; use_all_tfidf/
    # use_student_tfidf let us drop it and keep only the provider-INVARIANT
    # objective-text TF-IDF (which transfers best across a domain shift).
    v_all = None
    parts = []
    if cfg.get("use_all_tfidf", True):
        v_all = _mk_tfidf(cfg["tfidf_all"]).fit(X_df["text_all"].fillna(""))
        parts.append(v_all.transform(X_df["text_all"].fillna("")))
    v_student = None
    if cfg["use_student_tfidf"]:
        v_student = _mk_tfidf(cfg["tfidf_student"]).fit(X_df["text_student"].fillna(""))
        parts.append(v_student.transform(X_df["text_student"].fillna("")))
    v_lo = None
    if cfg["use_lo_tfidf"]:
        v_lo = _mk_tfidf(cfg["tfidf_lo"]).fit(X_df["text_lo"].fillna(""))
        parts.append(v_lo.transform(X_df["text_lo"].fillna("")))

    # --- LO target encoding ---
    enc_map = None
    lo_enc_col = None
    if cfg["use_lo_target_enc"]:
        obj = X_df["learning_objective_id"].astype(str).values
        enc_map = _lo_target_encode(obj, y, cfg["lo_smoothing"], global_mean)
        lo_enc_col = _apply_lo_enc(obj, enc_map, global_mean).reshape(-1, 1)

    # --- LR features: sparse tfidf + scaled numeric (+ lo enc) ---
    lr_blocks = list(parts) + [sparse.csr_matrix(num_scaled)]
    if lo_enc_col is not None:
        lr_blocks.append(sparse.csr_matrix(lo_enc_col))
    X_lr = sparse.hstack(lr_blocks).tocsr()
    lr = LogisticRegression(C=cfg["lr_C"], max_iter=2000, solver="liblinear")
    lr.fit(X_lr, y)

    # --- SVD of text for HGB ---
    svd = None
    svd_feat = None
    if cfg["svd_components"] and cfg["svd_components"] > 0 and parts:
        text_stack = sparse.hstack(parts).tocsr()
        k = min(cfg["svd_components"], text_stack.shape[1] - 1)
        svd = TruncatedSVD(n_components=k, random_state=0).fit(text_stack)
        svd_feat = svd.transform(text_stack)

    # --- HGB features: raw numeric (NaN ok) + svd + lo enc ---
    hgb_blocks = [num]  # keep NaN; HGB handles natively
    if svd_feat is not None:
        hgb_blocks.append(svd_feat)
    if lo_enc_col is not None:
        hgb_blocks.append(lo_enc_col)
    X_hgb = np.hstack(hgb_blocks)
    hgb = HistGradientBoostingClassifier(random_state=0, **cfg["hgb"])
    hgb.fit(X_hgb, y)

    return {
        "cfg": cfg, "num_cols": num_cols, "medians": medians, "scaler": scaler,
        "v_all": v_all, "v_student": v_student, "v_lo": v_lo,
        "svd": svd, "enc_map": enc_map, "global_mean": global_mean,
        "lr": lr, "hgb": hgb,
    }


def predict_pipeline(art, X_df):
    cfg = art["cfg"]
    num_cols = art["num_cols"]
    num = X_df[num_cols].to_numpy(dtype=float)
    num_imp = np.where(np.isnan(num), art["medians"], num)
    num_scaled = art["scaler"].transform(num_imp)

    parts = []
    if art["v_all"] is not None:
        parts.append(art["v_all"].transform(X_df["text_all"].fillna("")))
    if art["v_student"] is not None:
        parts.append(art["v_student"].transform(X_df["text_student"].fillna("")))
    if art["v_lo"] is not None:
        parts.append(art["v_lo"].transform(X_df["text_lo"].fillna("")))

    lo_enc_col = None
    if art["enc_map"] is not None:
        obj = X_df["learning_objective_id"].astype(str).values
        lo_enc_col = _apply_lo_enc(obj, art["enc_map"], art["global_mean"]).reshape(-1, 1)

    lr_blocks = list(parts) + [sparse.csr_matrix(num_scaled)]
    if lo_enc_col is not None:
        lr_blocks.append(sparse.csr_matrix(lo_enc_col))
    X_lr = sparse.hstack(lr_blocks).tocsr()
    p_lr = art["lr"].predict_proba(X_lr)[:, 1]

    hgb_blocks = [num]
    if art["svd"] is not None:
        text_stack = sparse.hstack(parts).tocsr()
        hgb_blocks.append(art["svd"].transform(text_stack))
    if lo_enc_col is not None:
        hgb_blocks.append(lo_enc_col)
    X_hgb = np.hstack(hgb_blocks)
    p_hgb = art["hgb"].predict_proba(X_hgb)[:, 1]

    w = cfg["blend_w_lr"]
    p = w * p_lr + (1 - w) * p_hgb
    # shrink toward the training prior — calibration insurance against the
    # train->test distribution shift (test is near-noise; over-confidence is punished).
    a = cfg.get("shrink_a", 1.0)
    prior = art.get("global_mean", 0.5)
    # shrink_center lets us recenter predictions on the estimated TEST base rate
    # (~0.685 here, vs the train rate 0.7025 the model's outputs center on). When
    # shrink_center == prior this reduces exactly to a*p + (1-a)*prior (back-compat).
    center = cfg.get("shrink_center", prior)
    if a < 1.0 or center != prior:
        p = center + a * (p - prior)
    c = cfg.get("clip", 0.0)
    if c and c > 0:
        p = np.clip(p, c, 1 - c)
    return p, p_lr, p_hgb
