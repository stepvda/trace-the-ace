"""Decision-grade, seed-averaged A/B driver for the GPU (RunPod RTX 4090).

Runs the objective-centered 'history' representation (earlier on-objective attempts, Fable
cycle-1 idea) vs the recency-only champion, at the VALIDATED instrument config
(subset 10k, batch 8, max_len 512, 2 epochs — the setup whose control lands ~0.59-0.63,
so a healthy control AUC confirms the ruler discriminates). Both arms hold total word
budget at 360 (pure reallocation — the harsh test; in the container history is ADDITIVE).

Averages the history-control AUC delta over several seeds (single-seed fine-tune noise is
large on this near-noise task). Usage: python gpu_ab.py <seeds csv> <subset> <batch>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ab_representation import run, ROOT
import dl_common as D
from sklearn.model_selection import StratifiedGroupKFold

ARMS = {"control": (0, 180, 180), "history": (110, 125, 125)}  # (HISTORY, RELEVANT, RECENT)


def one_seed(seed, subset, max_len, batch, epochs):
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    l = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = l.loc[f.response_id, "is_correct"].values
    f = f.sample(n=subset, random_state=seed).reset_index(drop=True)
    tdir = os.path.join(ROOT, "data", "train_transcripts")
    y = f.y.to_numpy(int); groups = f.learning_objective_id.astype(str).to_numpy()
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(f, y, groups))
    print(f"[seed{seed}] subset={len(f)} objectives={len(set(groups))} train={len(tr)} val={len(va)}", flush=True)
    out = {}
    for name, (h, rel, rec) in ARMS.items():
        D.HISTORY_WORDS, D.RELEVANT_WORDS, D.RECENT_WORDS = h, rel, rec
        t0 = time.time()
        texts = D.build_texts(f, tdir, n_words=360, centered=True, proxy_tags=True)
        hh = float(np.mean(["History:" in t for t in texts]))
        auc, ll = run(texts, y, tr, va, max_len, batch, epochs, seed)
        out[name] = (auc, ll)
        print(f"  seed{seed} {name:8s} AUC={auc:.4f} ll={ll:.5f} hasHist={hh:.2f} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43"])]
    subset = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print(f"=== GPU A/B: history vs control | seeds={seeds} subset={subset} batch={batch} ===", flush=True)
    deltas_auc, deltas_ll = [], []
    for s in seeds:
        o = one_seed(s, subset, 512, batch, 2)
        d_auc = o["history"][0] - o["control"][0]
        d_ll = o["control"][1] - o["history"][1]   # positive = history lowers log loss (better)
        deltas_auc.append(d_auc); deltas_ll.append(d_ll)
        print(f"=== seed{s}: history-control  AUC {d_auc:+.4f}  logloss {d_ll:+.5f} ===", flush=True)
    m_auc, m_ll = float(np.mean(deltas_auc)), float(np.mean(deltas_ll))
    sd_auc = float(np.std(deltas_auc)) if len(deltas_auc) > 1 else 0.0
    print(f"\n==== MEAN over {seeds}:  AUC {m_auc:+.4f} (sd {sd_auc:.4f}, per-seed {[round(x,4) for x in deltas_auc]})"
          f"  logloss {m_ll:+.5f} ====", flush=True)
    if m_auc >= 0.005:
        print("  => ADOPT history (regen train_texts + rebuild container)", flush=True)
    elif m_auc <= -0.002:
        print("  => KEEP control (history not better)", flush=True)
    else:
        print("  => MARGINAL/INCONCLUSIVE (add more seeds)", flush=True)


if __name__ == "__main__":
    main()
