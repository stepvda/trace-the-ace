"""A/B: objective-relevant-segment selector — 'last' (last overlapping mention, the
validated default) vs 'best' (densest / highest-total-overlap segment). Fable measured the
main objective discussion at median position 0.14, so 'last' may miss it. Same model, same
objective-grouped split, same seed; only SEG_MODE changes. Reuses the ab_representation run().
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from ab_representation import run, ROOT

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "distilbert_adapted")


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
    print(f"subset={len(f)} objectives={len(set(groups))} train={len(tr)} val={len(va)}", flush=True)

    D.RELEVANT_WORDS, D.RECENT_WORDS = 180, 180
    out = {}
    for mode in ("last", "best"):
        D.SEG_MODE = mode
        t0 = time.time()
        texts = D.build_texts(f, tdir, n_words=360, centered=True, proxy_tags=True)
        print(f"[{mode}] built texts in {time.time()-t0:.0f}s "
              f"(has-Relevant={np.mean(['Relevant:' in t for t in texts]):.2f})", flush=True)
        auc, ll = run(texts, y, tr, va, max_len, batch, epochs, seed)
        out[mode] = (auc, ll)
        print(f"  {mode.upper():5s} AUC={auc:.4f} logloss={ll:.5f}", flush=True)

    (al, ll_l), (ab, ll_b) = out["last"], out["best"]
    print(f"\n=== best − last: AUC {ab-al:+.4f}, logloss {ll_l-ll_b:+.5f} ===", flush=True)
    print("  => SWITCH default to 'best' (regen train_texts + rebuild zip)" if ab > al + 0.003
          else "  => keep 'last' (best not clearly better)", flush=True)


if __name__ == "__main__":
    main()
