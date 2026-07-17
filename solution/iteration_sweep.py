"""Disciplined transfer-improvement iteration.

Fitness = out-of-domain log loss on a leave-one-domain-out split (style-domains),
at each variant's best single shrink. Lower = better transfer. This is the best
local proxy for the real provider shift; to avoid proxy-overfitting we (a) only test
MECHANISM-GROUNDED ideas and (b) later re-verify winners across multiple clusterings.

Run: python iteration_sweep.py [screen|verify]  (screen = 1 clustering; verify = 3)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import log_loss
import shift_proxy as S
from model import fit_pipeline, predict_pipeline, DEFAULT_CONFIG

def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))

LO_WORD = {"max_features": 3000, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True, "analyzer": "word"}
LO_WORD13 = {**LO_WORD, "ngram_range": [1, 3]}
LO_CHAR35 = {"max_features": 6000, "ngram_range": [3, 5], "min_df": 2, "sublinear_tf": True, "analyzer": "char_wb"}
LO_CHAR25 = {**LO_CHAR35, "ngram_range": [2, 5]}
BASE = {"use_all_tfidf": False, "use_student_tfidf": False, "use_lo_tfidf": True, "svd_components": 80}

VARIANTS = {
    "ROBUST(word1-2)":    {**BASE, "tfidf_lo": LO_WORD},
    "char3-5":            {**BASE, "tfidf_lo": LO_CHAR35},
    "char2-5":            {**BASE, "tfidf_lo": LO_CHAR25},
    "word1-3":            {**BASE, "tfidf_lo": LO_WORD13},
    "lr_w0.75":           {**BASE, "tfidf_lo": LO_WORD, "blend_w_lr": 0.75},
    "lr_w0.35":           {**BASE, "tfidf_lo": LO_WORD, "blend_w_lr": 0.35},
    "lr_C0.3":            {**BASE, "tfidf_lo": LO_WORD, "lr_C": 0.3},
    "lr_C3":              {**BASE, "tfidf_lo": LO_WORD, "lr_C": 3.0},
    "no_svd":             {**BASE, "tfidf_lo": LO_WORD, "svd_components": 0},
    "numeric_only":       {**BASE, "use_lo_tfidf": False, "svd_components": 0},
}
ALPHAS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]


def fitness(X, y, obj, patch, cluster_cfgs):
    """Mean-over-domains out-of-domain logloss at the best single shrink, averaged
    over cluster configs. Returns (fitness, best_alpha, const_ll)."""
    import shift_proxy as S
    per_alpha_all = {a: [] for a in ALPHAS}
    const_all = []
    for (num, svd, seed, k) in cluster_cfgs:
        dom = S.make_domains(num, svd, k=k, seed=seed)
        for d in sorted(set(dom)):
            tr, te = dom != d, dom == d
            if te.sum() < 100:
                continue
            art = fit_pipeline({**patch, "shrink_a": 1.0}, X.iloc[tr], y[tr])
            p, _, _ = predict_pipeline(art, X.iloc[te])
            prior = float(y[tr].mean())
            const_all.append(L(y[te], np.full(te.sum(), prior)))
            for a in ALPHAS:
                per_alpha_all[a].append(L(y[te], a * p + (1 - a) * prior))
    means = {a: float(np.mean(v)) for a, v in per_alpha_all.items()}
    ba = min(means, key=means.get)
    return means[ba], ba, float(np.mean(const_all))


def main(mode="screen"):
    X, num, svd, y, obj = S.load()
    if mode == "verify":
        cfgs = [(num, svd, s, k) for s, k in [(0, 6), (1, 8), (2, 5)]]
    else:
        cfgs = [(num, svd, 0, 6)]
    print(f"mode={mode}  n={len(y)}  cluster_cfgs={len(cfgs)}\n", flush=True)
    rows = []
    t0 = time.time()
    for name, patch in VARIANTS.items():
        t = time.time()
        fit, ba, const = fitness(X, y, obj, patch, cfgs)
        rows.append((name, fit, ba, const - fit))
        print(f"  {name:<18} transfer_ll={fit:.4f}  best_a={ba:.2f}  gain_vs_const={const-fit:+.4f}  "
              f"({time.time()-t:.0f}s)", flush=True)
    rows.sort(key=lambda r: r[1])
    print(f"\n=== RANKED by transfer logloss (lower=better), total {time.time()-t0:.0f}s ===")
    for name, fit, ba, g in rows:
        print(f"  {fit:.4f}  a={ba:.2f}  gain={g:+.4f}  {name}")
    print(f"\nbest: {rows[0][0]}  (constant≈{rows[0][1]+rows[0][3]:.4f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "screen")
