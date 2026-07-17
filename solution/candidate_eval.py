"""Decisive validation of the transfer-robust candidate vs the current FULL model,
using the REAL pipeline (fit_pipeline/predict_pipeline), refit leak-free.

Configs:
  FULL    = current shipped model (all-transcript + student + objective TF-IDF + numeric)
  ROBUST  = drop transcript TF-IDF (all+student), keep objective-text TF-IDF + numeric
  ROBUST+ = ROBUST but student kept (drop only full-transcript all)

Metrics:
  (1) objective-grouped OOF  -> honest IN-DOMAIN cost of dropping transcript text
  (2) style-domain leave-one-out -> TRANSFER (the real target); with shrink sweep
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import shift_proxy as S
from model import fit_pipeline, predict_pipeline, DEFAULT_CONFIG

def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))

# Memory-feasible TF-IDF on the 8GB M1 (the FULL-vs-ROBUST *comparison* is invariant
# to exact vocab size; full 30k vocab thrashes swap). Applied to every config.
_LEAN = {
    "tfidf_all": {"max_features": 8000, "ngram_range": [1, 2], "min_df": 5, "sublinear_tf": True},
    "tfidf_student": {"max_features": 5000, "ngram_range": [1, 2], "min_df": 5, "sublinear_tf": True},
    "tfidf_lo": {"max_features": 3000, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
    "svd_components": 80,
}
CONFIGS = {
    "FULL":    {**_LEAN, "use_all_tfidf": True,  "use_student_tfidf": True,  "use_lo_tfidf": True},
    "ROBUST":  {**_LEAN, "use_all_tfidf": False, "use_student_tfidf": False, "use_lo_tfidf": True},
    "ROBUST+": {**_LEAN, "use_all_tfidf": False, "use_student_tfidf": True,  "use_lo_tfidf": True},
}


def objgrouped(X, y, obj, cfgpatch):
    oof = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, obj):
        art = fit_pipeline({**cfgpatch, "shrink_a": 1.0}, X.iloc[tr], y[tr])
        oof[va], _, _ = predict_pipeline(art, X.iloc[va])
    return oof


def domain_transfer(X, y, dom, cfgpatch, alphas=(0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)):
    """Leave-one-domain-out; return per-alpha mean logloss + mean gain-vs-constant at best alpha."""
    per_alpha = {a: [] for a in alphas}
    for d in sorted(set(dom)):
        tr, te = dom != d, dom == d
        if te.sum() < 100:
            continue
        art = fit_pipeline({**cfgpatch, "shrink_a": 1.0}, X.iloc[tr], y[tr])
        p, _, _ = predict_pipeline(art, X.iloc[te])
        prior = float(y[tr].mean())
        for a in alphas:
            ps = a * p + (1 - a) * prior
            per_alpha[a].append(L(y[te], ps))
    means = {a: float(np.mean(v)) for a, v in per_alpha.items()}
    best_a = min(means, key=means.get)
    return means, best_a


def main():
    X, num, svd, y, obj = S.load()
    dom = S.make_domains(num, svd, k=5)
    const_dom = float(np.mean([L(y[dom == d], np.full((dom == d).sum(), y[dom != d].mean()))
                               for d in sorted(set(dom)) if (dom == d).sum() >= 100]))
    print(f"n={len(y)} | style-domain constant baseline logloss={const_dom:.4f}\n")
    print(f"{'config':<9} {'objgrp_ll':>9} {'objgrp_auc':>10} | {'transfer_ll@best_a':>18} {'best_a':>6} {'gain_vs_const':>13}")
    for name, patch in CONFIGS.items():
        oof = objgrouped(X, y, obj, patch)
        og_ll, og_auc = L(y, oof), roc_auc_score(y, oof)
        means, ba = domain_transfer(X, y, dom, patch)
        print(f"{name:<9} {og_ll:>9.4f} {og_auc:>10.4f} | {means[ba]:>18.4f} {ba:>6.1f} "
              f"{const_dom - means[ba]:>+13.4f}", flush=True)


if __name__ == "__main__":
    main()
