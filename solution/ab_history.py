"""A/B (cycle 1): 'history' mode — does prepending the student's EARLIER on-objective
attempts (chronological, under their own budget) beat the recency-only champion?

- Arm A (control): HISTORY_WORDS=0, RELEVANT_WORDS=180, RECENT_WORDS=180 — byte-identical
  to the shipped champion texts.
- Arm B (history): HISTORY_WORDS=110, RELEVANT_WORDS=125, RECENT_WORDS=125.

Total word budget held at 360 in BOTH arms, so the comparison is pure REALLOCATION, not
more tokens (the harsher test: at 512 tokens the proxy must give up recency budget to fit
history; ModernBERT at 3072 can ADD history on top — so a proxy win understates the
container gain). Same model, same objective-grouped split, same seed. Reuses ab_representation.run().

Decision: adopt if AUC delta >= +0.005; if |delta| < 0.010, rerun both arms at seed 43 and
average the two deltas (single-seed MPS noise ~±0.03 warrants it for small effects).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from ab_representation import run, ROOT


def main(subset=10000, max_len=512, batch=8, epochs=2, seed=42):
    import dl_common as D
    from sklearn.model_selection import StratifiedGroupKFold
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    l = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = l.loc[f.response_id, "is_correct"].values
    f = f.sample(n=subset, random_state=seed).reset_index(drop=True)
    tdir = os.path.join(ROOT, "data", "train_transcripts")
    y = f.y.to_numpy(int); groups = f.learning_objective_id.astype(str).to_numpy()
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(f, y, groups))
    print(f"subset={len(f)} objectives={len(set(groups))} train={len(tr)} val={len(va)} seed={seed}", flush=True)

    arms = {
        "control": dict(HISTORY_WORDS=0,   RELEVANT_WORDS=180, RECENT_WORDS=180),
        "history": dict(HISTORY_WORDS=110, RELEVANT_WORDS=125, RECENT_WORDS=125),
    }
    out = {}
    for name, cfg in arms.items():
        D.HISTORY_WORDS = cfg["HISTORY_WORDS"]
        D.RELEVANT_WORDS = cfg["RELEVANT_WORDS"]
        D.RECENT_WORDS = cfg["RECENT_WORDS"]
        t0 = time.time()
        texts = D.build_texts(f, tdir, n_words=360, centered=True, proxy_tags=True)
        has_hist = float(np.mean(["History:" in t for t in texts]))
        print(f"[{name}] built in {time.time()-t0:.0f}s has-History={has_hist:.2f}", flush=True)
        auc, ll = run(texts, y, tr, va, max_len, batch, epochs, seed)
        out[name] = (auc, ll)
        print(f"  {name.upper():8s} AUC={auc:.4f} logloss={ll:.5f}", flush=True)

    (ac, llc), (ah, llh) = out["control"], out["history"]
    d = ah - ac
    print(f"\n=== history - control: AUC {d:+.4f}, logloss {llc-llh:+.5f} ===", flush=True)
    if d >= 0.005:
        print("  => ADOPT history (regen train_texts + rebuild zip)", flush=True)
    elif abs(d) < 0.010:
        print("  => INCONCLUSIVE (|delta|<0.010) — rerun at seed 43 and average deltas", flush=True)
    else:
        print("  => KEEP control (history clearly not better)", flush=True)


if __name__ == "__main__":
    main()
