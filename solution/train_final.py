"""Fit the final pipeline on ALL training data and save artifacts for inference.

Writes: submission/assets/artifacts.pkl
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import joblib
from model import fit_pipeline, DEFAULT_CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ASSETS = os.path.join(ROOT, "submission", "assets")
os.makedirs(ASSETS, exist_ok=True)


def main(config_path=None):
    cfg = None
    if config_path and os.path.exists(config_path):
        cfg = json.load(open(config_path))
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[X.index, "is_correct"].to_numpy(dtype=float)
    print(f"fitting final pipeline on {X.shape[0]} rows ...", flush=True)
    t0 = time.time()
    art = fit_pipeline(cfg, X, y)
    print(f"fit done in {time.time()-t0:.1f}s. config blend_w_lr={art['cfg']['blend_w_lr']}", flush=True)
    out = os.path.join(ASSETS, "artifacts.pkl")
    joblib.dump(art, out, compress=3)
    sz = os.path.getsize(out) / 1e6
    print(f"saved artifacts -> {out} ({sz:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
