"""Fable Round-4 Part A (CPU, $0): heterogeneity read of the transformer blend's gain across
STYLE domains, on the CACHED OOF arrays (no retraining). If the blend's gain over classical is
roughly uniform across style clusters -> style-independent -> likely transfers (LB bands 1-2).
If one cluster carries most of the gain -> style-fragile -> expect band 3.

Both legs are Platt-calibrated (removes calibration, isolates DISCRIMINATION, which is what
survives to the LB). Reports per-domain calLL gain and AUROC gain.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from shift_proxy import make_domains

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def sig(z): return 1 / (1 + np.exp(-z))
def L(y, p): return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
def platt(p, y): return LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y).predict_proba(logit(p).reshape(-1, 1))[:, 1]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ids = pd.read_csv(os.path.join(CACHE, "row_ids.csv")).iloc[:, 0].astype(str).tolist()
    num = np.load(os.path.join(CACHE, "num.npy")); svd = np.load(os.path.join(CACHE, "svd256.npy"))
    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med); num = np.where(np.isnan(num), med, num)
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[ids, "is_correct"].to_numpy(float)

    cls = pd.read_parquet(os.path.join(ROOT, "submission", "assets", "classical_oof.parquet")).set_index("response_id")
    t1 = pd.read_parquet(os.path.join(ROOT, "oof_transformer.parquet")).set_index("response_id")
    t2 = pd.read_parquet(os.path.join(ROOT, "oof_control_s2.parquet")).set_index("response_id")
    pc = cls.loc[ids, "p_classical"].to_numpy(float)
    ptr = ((t1.p_control.reindex(ids) + t2.p_control.reindex(ids)) / 2).to_numpy(float)
    bc = json.load(open(os.path.join(ROOT, "submission", "assets", "blend_config.json")))
    w = bc["blend_w"]

    p_cls = platt(pc, y)                                   # calibrated classical
    p_bl = platt(w * ptr + (1 - w) * pc, y)                # calibrated blend
    dom = make_domains(num, svd, k=K)

    print(f"K={K} | overall: cls calLL {L(y,p_cls):.5f}  blend calLL {L(y,p_bl):.5f}  "
          f"gain {L(y,p_cls)-L(y,p_bl):+.5f}  | cls AUROC {roc_auc_score(y,p_cls):.4f} blend {roc_auc_score(y,p_bl):.4f}", flush=True)
    print(f"{'dom':>3} {'n':>6} {'rate':>6} {'clsLL':>8} {'blendLL':>8} {'LLgain':>8} {'clsAUC':>7} {'blAUC':>7} {'AUCgain':>8}")
    rows = []
    for d in sorted(set(dom)):
        m = dom == d
        if m.sum() < 50 or len(set(y[m])) < 2:
            continue
        lg = L(y[m], p_cls[m]) - L(y[m], p_bl[m])
        ac, ab = roc_auc_score(y[m], p_cls[m]), roc_auc_score(y[m], p_bl[m])
        rows.append((int(m.sum()), lg, ab - ac))
        print(f"{d:>3} {m.sum():>6} {y[m].mean():>6.3f} {L(y[m],p_cls[m]):>8.4f} {L(y[m],p_bl[m]):>8.4f} "
              f"{lg:>+8.4f} {ac:>7.4f} {ab:>7.4f} {ab-ac:>+8.4f}", flush=True)
    n = np.array([r[0] for r in rows]); llg = np.array([r[1] for r in rows]); aucg = np.array([r[2] for r in rows])
    print(f"\nLL gain across domains:   mean {llg.mean():+.5f}  min {llg.min():+.5f}  max {llg.max():+.5f}  "
          f">0 in {(llg>0).sum()}/{len(llg)} domains", flush=True)
    print(f"AUROC gain across domains: mean {aucg.mean():+.5f}  min {aucg.min():+.5f}  max {aucg.max():+.5f}  "
          f">0 in {(aucg>0).sum()}/{len(aucg)} domains", flush=True)
    # concentration: fraction of total LL gain from the single best domain
    tot = float(np.sum(n * llg)); best = float(np.max(n * llg))
    print(f"CONCENTRATION: best domain contributes {best/tot*100:.0f}% of total LL gain "
          f"({'UNIFORM -> likely transfers (bands 1-2)' if (aucg>0).mean()>=0.8 and best/tot<0.5 else 'CONCENTRATED/MIXED -> style-fragile, expect band 3'})", flush=True)


if __name__ == "__main__":
    main()
