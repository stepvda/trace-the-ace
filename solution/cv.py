"""Cross-validated evaluation (StratifiedGroupKFold by session) for a config.

Usage: python cv.py [config.json]   (config optional; merged over DEFAULT_CONFIG)
Prints OOF log loss / AUC for LR, HGB, and the blend.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from model import fit_pipeline, predict_pipeline, DEFAULT_CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load():
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[X.index, "is_correct"].to_numpy(dtype=float)
    groups = f.loc[X.index, "session_id"].astype(str).to_numpy()
    return X, y, groups


def run_cv(config=None, n_splits=5, verbose=True):
    X, y, groups = load()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y)); oof_lr = np.zeros(len(y)); oof_hgb = np.zeros(len(y))
    t0 = time.time()
    for i, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        art = fit_pipeline(config, X.iloc[tr], y[tr])
        p, plr, phgb = predict_pipeline(art, X.iloc[va])
        oof[va], oof_lr[va], oof_hgb[va] = p, plr, phgb
        if verbose:
            print(f"  fold{i}: blend={log_loss(y[va], np.clip(p,1e-6,1-1e-6)):.5f} "
                  f"lr={log_loss(y[va], np.clip(plr,1e-6,1-1e-6)):.5f} "
                  f"hgb={log_loss(y[va], np.clip(phgb,1e-6,1-1e-6)):.5f}", flush=True)
    res = {
        "blend_ll": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        "lr_ll": float(log_loss(y, np.clip(oof_lr, 1e-6, 1 - 1e-6))),
        "hgb_ll": float(log_loss(y, np.clip(oof_hgb, 1e-6, 1 - 1e-6))),
        "blend_auc": float(roc_auc_score(y, oof)),
        "secs": round(time.time() - t0, 1),
    }
    # search best static blend weight on OOF
    best_w, best_ll = res["blend_w_lr"] if "blend_w_lr" in res else 0.5, res["blend_ll"]
    for w in np.linspace(0, 1, 21):
        pll = log_loss(y, np.clip(w * oof_lr + (1 - w) * oof_hgb, 1e-6, 1 - 1e-6))
        if pll < best_ll:
            best_ll, best_w = pll, w
    res["best_blend_w_lr"] = float(best_w)
    res["best_blend_ll"] = float(best_ll)
    return res


if __name__ == "__main__":
    cfg = None
    if len(sys.argv) > 1:
        cfg = json.load(open(sys.argv[1]))
    print("config override:", cfg)
    res = run_cv(cfg)
    print("RESULT", json.dumps(res))
    print(f"baseline(const) logloss = 0.60876 | blend OOF = {res['blend_ll']:.5f} "
          f"(lr {res['lr_ll']:.5f}, hgb {res['hgb_ll']:.5f}) | AUC {res['blend_auc']:.4f}")
    print(f"best static blend w_lr={res['best_blend_w_lr']:.2f} -> {res['best_blend_ll']:.5f}")
