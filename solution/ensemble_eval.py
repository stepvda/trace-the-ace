"""Evaluate whether the transformer OOF adds value on top of the classical OOF,
on the SAME objective-grouped held-out responses (leakage-free). Reports each
model's log loss/AUC, their correlation (decorrelation => ensemble helps), and
the best blend weight + resulting metrics.

Usage: python ensemble_eval.py [ft_val_tag]   (default electra_v1)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "submission", "assets")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def main(tag="electra_v1"):
    cls = pd.read_parquet(os.path.join(ASSETS, "classical_oof.parquet")).set_index("response_id")
    ft = pd.read_csv(os.path.join(CACHE, f"ft_val_{tag}.csv")).set_index("response_id")
    # align on the transformer's held-out responses
    j = ft.join(cls[["p_classical"]], how="inner").dropna()
    y = j["y"].to_numpy(float)
    pc = j["p_classical"].to_numpy(float)
    pf = j["p_ft"].to_numpy(float)
    print(f"aligned {len(j)} held-out responses (objective-grouped)")
    print(f"classical: ll={L(y,pc):.5f} auc={roc_auc_score(y,pc):.4f}")
    print(f"transformer: ll={L(y,pf):.5f} auc={roc_auc_score(y,pf):.4f}")
    print(f"pred correlation (lower=more decorrelated): {np.corrcoef(pc,pf)[0,1]:.3f}")
    best_w, best_ll = 0.0, L(y, pc)
    for w in np.linspace(0, 1, 41):
        ll = L(y, (1 - w) * pc + w * pf)
        if ll < best_ll:
            best_ll, best_w = ll, w
    pe = (1 - best_w) * pc + best_w * pf
    print(f"BEST blend: w_transformer={best_w:.3f} -> ll={best_ll:.5f} auc={roc_auc_score(y,pe):.4f}")
    print(f"improvement over classical: {L(y,pc)-best_ll:+.5f} logloss, "
          f"{roc_auc_score(y,pe)-roc_auc_score(y,pc):+.4f} auc")
    if best_w < 0.02:
        print("=> transformer does NOT help; keep classical.")
    else:
        print("=> transformer ADDS value; bundle a trained-on-all transformer + ensemble.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "electra_v1")
