"""Local (no-GPU) blend + gate: combine the transformer OOF(s) (gpu_oof.py) with the classical
OOF (submission/assets/classical_oof.parquet). Accepts MULTIPLE transformer OOF parquets and
seed-averages each rep's p_<rep> across them (Fable's 2-seed confirmation). For each variant
{control, history, both}, fit the blend weight + a final Platt calibration on OOF, and evaluate
the PRE-COMMITTED gate:
    blended OOF AUROC >= classical + AUC_GATE  AND  calibrated OOF logloss <= classical - LL_GATE.
Prints the winning variant + blend weight + Platt (A,B) constants and writes blend_config.json.
Usage: python blend_gate.py <oof1.parquet> [oof2.parquet ...]
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUC_GATE = 0.008   # Fable's revised gate (complementary-leg regime): AUROC >= classical + 0.008
LL_GATE = 0.004    #                                                  calLL <= classical - 0.004


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    paths = sys.argv[1:] or [os.path.join(ROOT, "oof_transformer.parquet")]
    cls = pd.read_parquet(os.path.join(ROOT, "submission", "assets", "classical_oof.parquet")).set_index("response_id")
    y = cls["y"].to_numpy(float); pc = cls["p_classical"].to_numpy(float)
    base_auc = roc_auc_score(y, pc); base_ll = log_loss(y, np.clip(pc, 1e-6, 1 - 1e-6))
    print(f"gate: AUROC>=+{AUC_GATE} AND calLL<=-{LL_GATE} | files={len(paths)}")
    print(f"n={len(cls)}  CLASSICAL: AUROC={base_auc:.4f} logloss={base_ll:.5f}")

    # seed-average each rep's p_<rep> across all provided OOF files (aligned by response_id)
    rep_arrs = {}
    for path in paths:
        t = pd.read_parquet(path).set_index("response_id")
        for c in t.columns:
            if c.startswith("p_"):
                rep_arrs.setdefault(c[2:], []).append(t[c].reindex(cls.index))
    variants = {}
    for rep, arrs in rep_arrs.items():
        variants[rep] = (sum(arrs) / len(arrs)).to_numpy(float)
        print(f"  rep '{rep}': seed-averaged over {len(arrs)} file(s)")
    reps = list(variants.keys())
    if len(reps) > 1:
        variants["both"] = np.mean([variants[r] for r in reps], axis=0)

    results = []
    for name, pt in variants.items():
        auc_t = roc_auc_score(y, pt)
        best_w, best_ll = 0.0, base_ll
        for w in np.linspace(0, 1, 51):
            pb = w * pt + (1 - w) * pc
            ll = log_loss(y, np.clip(pb, 1e-6, 1 - 1e-6))
            if ll < best_ll:
                best_ll, best_w = ll, w
        pb = best_w * pt + (1 - best_w) * pc
        auc_b = roc_auc_score(y, pb)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pb).reshape(-1, 1), y)
        A, B = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        pcal = lr.predict_proba(logit(pb).reshape(-1, 1))[:, 1]
        ll_cal = log_loss(y, np.clip(pcal, 1e-6, 1 - 1e-6))
        gate = (auc_b >= base_auc + AUC_GATE) and (ll_cal <= base_ll - LL_GATE)
        results.append((name, best_w, auc_t, auc_b, ll_cal, A, B, gate, float(np.mean(pcal))))
        print(f"  {name:8s}: transAUROC={auc_t:.4f} | w={best_w:.2f} blendAUROC={auc_b:.4f} "
              f"(+{auc_b-base_auc:.4f}) calLL={ll_cal:.5f} ({ll_cal-base_ll:+.5f}) "
              f"platt(A={A:.3f},B={B:.3f}) GATE={'PASS' if gate else 'fail'}")

    passed = [r for r in results if r[7]]
    if passed:
        win = min(passed, key=lambda r: r[4])
        active = reps if win[0] == "both" else [win[0]]
        cfg = {"variant": win[0], "active_reps": active, "blend_w": round(win[1], 4),
               "platt_A": round(win[5], 5), "platt_B": round(win[6], 5),
               "blend_pivot": round(win[8], 5),   # post-Platt blend OOF mean -> recenter pivot
               "recenter_center": 0.685, "recenter_a": 0.68,  # LB-validated affine (Fable)
               "classical_auc": round(base_auc, 4), "classical_ll": round(base_ll, 5),
               "blend_auc": round(win[3], 4), "cal_ll": round(win[4], 5), "n_seeds": len(paths)}
        out = os.path.join(ROOT, "submission", "assets", "blend_config.json")
        json.dump(cfg, open(out, "w"), indent=2)
        print(f"\n>>> SHIP variant='{win[0]}' reps={active} blend_w={win[1]:.3f} "
              f"(calLL {win[4]:.5f} vs classical {base_ll:.5f}) <<<")
        print("saved", out, cfg)
    else:
        print("\n>>> NO variant passes the gate — DROP the transformer, keep classical 0.6087 <<<")


if __name__ == "__main__":
    main()
