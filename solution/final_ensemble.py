"""Combine all available model OOF (classical variants + transformer views) on the
SAME objective-grouped held-out responses and find a robust blend via GREEDY
forward selection (avoids overfitting a meta-learner on the small val set).
Reports the ensemble vs. classical baseline.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "submission", "assets")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def load_all():
    cols = {}
    cls = pd.read_parquet(os.path.join(ASSETS, "classical_oof.parquet")).set_index("response_id")
    cols["classical"] = cls["p_classical"]; y = cls["y"]
    for p in glob.glob(os.path.join(CACHE, "oof_*.parquet")):
        name = os.path.basename(p)[4:-8]
        cols[name] = pd.read_parquet(p).set_index("response_id")["p"]
    for p in glob.glob(os.path.join(CACHE, "ft_val_*.csv")):
        name = os.path.basename(p)[7:-4]
        cols["ft_" + name] = pd.read_csv(p).set_index("response_id")["p_ft"]
    df = pd.DataFrame(cols)
    df["y"] = y
    return df.dropna()


def greedy_blend(df, models, y):
    # start from classical; add fractional weights of models that reduce val logloss
    ens = df["classical"].to_numpy(float).copy()
    weights = {"classical": 1.0}
    improved = True
    step = 0.1
    while improved:
        improved = False
        best = None
        for m in models:
            pm = df[m].to_numpy(float)
            cand = (1 - step) * ens + step * pm
            ll = L(y, cand)
            if best is None or ll < best[1]:
                best = (m, ll, cand)
        if best and best[1] < L(y, ens) - 1e-5:
            ens = best[2]; weights[best[0]] = weights.get(best[0], 0) + step; improved = True
    return ens, weights


def main():
    df = load_all()
    y = df["y"].to_numpy(float)
    models = [c for c in df.columns if c != "y"]
    print(f"aligned {len(df)} held-out responses; models: {models}")
    for m in models:
        p = df[m].to_numpy(float)
        print(f"  {m:16s} ll={L(y,p):.5f} auc={roc_auc_score(y,p):.4f}")
    base = L(y, df["classical"].to_numpy(float))
    ens, w = greedy_blend(df, [m for m in models if m != "classical"], y)
    print(f"\nGREEDY ENSEMBLE ll={L(y,ens):.5f} auc={roc_auc_score(y,ens):.4f}")
    print(f"classical baseline ll={base:.5f} | improvement {base-L(y,ens):+.5f}")
    print("weights:", {k: round(v, 2) for k, v in w.items() if v > 0})


if __name__ == "__main__":
    main()
