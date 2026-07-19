"""Fable Round-4 Part B: leave-one-STYLE-domain-out for the transformer control leg.

For each of K style domains (data/domain_assignment.csv), train ModernBERT-base (control rep, the
shipped representation) on all OTHER domains and predict the HELD-OUT domain -> out-of-domain
transformer predictions on transcripts whose STYLE the model never saw. Blended with the classical
LODO locally to read whether the leg's gain survives a style shift (the closest local analog of the
LB provider shift). Saves /workspace/oof_lodo.parquet. Usage: python gpu_lodo.py <epochs>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from gpu_oof import train_predict          # identical training to the base OOF
from gpu_mbert import build_texts_for, ARMS, ROOT, BASE


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = lab.loc[f.response_id, "is_correct"].values
    dom = pd.read_csv(os.path.join(ROOT, "data", "domain_assignment.csv")).set_index("response_id")
    f["domain"] = dom.loc[f.response_id, "domain"].values
    y = f.y.to_numpy(int); rid = f.response_id.astype(str).to_numpy()
    fold_dom = f.domain.to_numpy()
    print(f"=== LODO control leg | base={BASE} epochs={epochs} domains={sorted(set(fold_dom))} ===", flush=True)

    rep = "control"; cfg = ARMS[rep]
    texts = build_texts_for(f, rep)
    oof = np.zeros(len(f)); t0 = time.time()
    for d in sorted(set(fold_dom)):
        va = np.where(fold_dom == d)[0]; tr = np.where(fold_dom != d)[0]
        ts = time.time()
        vp = train_predict(rep, texts, y, tr, va, cfg["max_len"], cfg["batch"], cfg["accum"], epochs, seed=42 + int(d))
        oof[va] = vp
        print(f"[domain{d}] OUT-OF-STYLE val_auc={roc_auc_score(y[va], vp):.4f} n_va={len(va)} n_tr={len(tr)} "
              f"({int(time.time()-ts)}s, tot {int(time.time()-t0)}s)", flush=True)
    pd.DataFrame({"response_id": rid, "y": y, "domain": fold_dom, "p_control_lodo": oof}).to_parquet("/workspace/oof_lodo.parquet")
    print(f"=== OOF control_lodo (pooled) AUROC={roc_auc_score(y, oof):.4f} ===", flush=True)
    print("LODO_DONE saved /workspace/oof_lodo.parquet", flush=True)


if __name__ == "__main__":
    main()
