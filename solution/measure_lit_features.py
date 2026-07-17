"""Measure the incremental value of the literature-grounded features via a
low-memory, numeric-only objective-grouped A/B (HGB): with vs. without the 10
new features. Reports AUC / log loss deltas.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from features import numeric_feature_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# all v3 features (to exclude for the base model)
V3_ALL = ["proxy_acc", "proxy_last", "proxy_lastk", "proxy_recency_acc", "proxy_trail_correct",
          "proxy_trail_incorrect", "proxy_n_correct", "proxy_n_incorrect", "proxy_n_answers",
          "proxy_correct_minus_incorrect", "rapid_guess_frac", "fb_process", "fb_selfreg",
          "fb_person", "fb_highinfo_share", "fb_person_share", "telling", "telling_share",
          "obj_ntokens", "obj_mean_wordlen", "obj_symbolic", "obj_multistep",
          "obj_coverage_all", "obj_coverage_student"]
# refined high-signal subset (tightened proxy + difficulty; drop skewed/collinear/noisy)
REFINED = ["proxy_last", "proxy_lastk", "proxy_recency_acc", "proxy_acc", "proxy_trail_incorrect",
           "proxy_correct_minus_incorrect", "rapid_guess_frac", "fb_highinfo_share",
           "obj_symbolic", "obj_multistep", "obj_coverage_student"]
LIT = V3_ALL


def run(X, cols, y, groups):
    oof = np.zeros(len(y))
    M = X[cols].to_numpy(float)
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(M, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, random_state=0)
        clf.fit(M[tr], y[tr]); oof[va] = clf.predict_proba(M[va])[:, 1]
    return float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))), float(roc_auc_score(y, oof))


def main():
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    y = lab.loc[X.index, "is_correct"].to_numpy(float)
    groups = f.loc[X.index, "learning_objective_id"].astype(str).to_numpy()
    num = numeric_feature_columns(X)
    base = [c for c in num if c not in V3_ALL]
    refined = base + [c for c in REFINED if c in num]
    full = num
    print(f"numeric-only HGB, objective-grouped (base={len(base)}, refined={len(refined)}, full={len(full)}):", flush=True)
    bll, bauc = run(X, base, y, groups)
    rll, rauc = run(X, refined, y, groups)
    fll, fauc = run(X, full, y, groups)
    print(f"  base (talk-moves)      : ll={bll:.5f} auc={bauc:.4f}")
    print(f"  + refined proxy subset : ll={rll:.5f} auc={rauc:.4f}   effect {rauc-bauc:+.4f} AUC, {bll-rll:+.5f} ll")
    print(f"  + full v3 batch        : ll={fll:.5f} auc={fauc:.4f}   effect {fauc-bauc:+.4f} AUC, {bll-fll:+.5f} ll")
    print("  => REFINED subset HELPS" if rauc > bauc + 0.001 else "  => refined subset no clear gain")


if __name__ == "__main__":
    main()
