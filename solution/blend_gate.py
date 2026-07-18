"""Local (no-GPU) blend + gate: combine the transformer OOF (gpu_oof.py) with the classical OOF
(submission/assets/classical_oof.parquet), for each variant {control, history, both}, fit the blend
weight + a final Platt calibration on OOF, and evaluate Fable's ship gate:
    blended OOF AUROC >= classical + 0.015  AND  calibrated OOF logloss <= classical - 0.002.
Prints the winning variant + blend weight + Platt (A,B) constants to bake into the container.
Usage: python blend_gate.py <oof_transformer.parquet>
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    oof_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "oof_transformer.parquet")
    cls = pd.read_parquet(os.path.join(ROOT, "submission", "assets", "classical_oof.parquet"))
    tr = pd.read_parquet(oof_path)
    d = cls.merge(tr.drop(columns=[c for c in ("y",) if c in tr.columns]), on="response_id")
    y = d["y"].to_numpy(float)
    pc = d["p_classical"].to_numpy(float)
    base_auc = roc_auc_score(y, pc); base_ll = log_loss(y, np.clip(pc, 1e-6, 1 - 1e-6))
    print(f"n={len(d)}  CLASSICAL: AUROC={base_auc:.4f} logloss={base_ll:.5f}")

    reps = [c[2:] for c in d.columns if c.startswith("p_") and c != "p_classical"]
    variants = {r: d["p_" + r].to_numpy(float) for r in reps}
    if len(reps) > 1:
        variants["both"] = np.mean([d["p_" + r].to_numpy(float) for r in reps], axis=0)
        print(f"transformer legs: {reps} (+ 'both' = their mean)")

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
        gate = (auc_b >= base_auc + 0.015) and (ll_cal <= base_ll - 0.002)
        results.append((name, best_w, auc_t, auc_b, ll_cal, A, B, gate))
        print(f"  {name:8s}: transAUROC={auc_t:.4f} | w={best_w:.2f} blendAUROC={auc_b:.4f} "
              f"(+{auc_b-base_auc:.4f}) calLL={ll_cal:.5f} ({ll_cal-base_ll:+.5f}) "
              f"platt(A={A:.3f},B={B:.3f}) GATE={'PASS' if gate else 'fail'}")

    passed = [r for r in results if r[7]]
    if passed:
        win = min(passed, key=lambda r: r[4])  # lowest calibrated logloss
        active = reps if win[0] == "both" else [win[0]]
        cfg = {"variant": win[0], "active_reps": active, "blend_w": round(win[1], 4),
               "platt_A": round(win[5], 5), "platt_B": round(win[6], 5),
               "classical_auc": round(base_auc, 4), "classical_ll": round(base_ll, 5),
               "blend_auc": round(win[3], 4), "cal_ll": round(win[4], 5)}
        out = os.path.join(ROOT, "submission", "assets", "blend_config.json")
        json.dump(cfg, open(out, "w"), indent=2)
        print(f"\n>>> SHIP variant='{win[0]}' reps={active} blend_w={win[1]:.3f} "
              f"platt_A={win[5]:.4f} platt_B={win[6]:.4f} (calLL {win[4]:.5f} vs classical {base_ll:.5f}) <<<")
        print("saved", out, cfg)
    else:
        print("\n>>> NO variant passes the gate — do NOT ship the transformer (red flag; investigate) <<<")


if __name__ == "__main__":
    main()
