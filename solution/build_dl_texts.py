"""Precompute the transformer training texts locally and bundle them (compact),
so the container doesn't need the 576MB raw transcripts to train the DL model.
Saves submission/assets/train_texts.parquet with columns:
  response_id, text, y, learning_objective_id
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from dl_common import build_texts, DEFAULT_N_WORDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "submission", "assets")
os.makedirs(ASSETS, exist_ok=True)


def main(n_words=DEFAULT_N_WORDS):
    feats = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    labs = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    t0 = time.time()
    texts = build_texts(feats, os.path.join(ROOT, "data", "train_transcripts"), n_words=n_words)
    df = pd.DataFrame({
        "response_id": feats.response_id.values,
        "text": texts,
        "y": labs.loc[feats.response_id, "is_correct"].values,
        "learning_objective_id": feats.learning_objective_id.astype(str).values,
    })
    out = os.path.join(ASSETS, "train_texts.parquet")
    df.to_parquet(out)
    sz = os.path.getsize(out) / 1e6
    print(f"built {len(df)} texts in {time.time()-t0:.1f}s -> {out} ({sz:.1f} MB); "
          f"avg words={df.text.str.split().apply(len).mean():.0f}")


if __name__ == "__main__":
    main()
