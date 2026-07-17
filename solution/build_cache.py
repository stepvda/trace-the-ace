"""Build the engineered feature table for all training responses once and cache it.

Saves numeric features + text columns to solution/cache so model experiments
don't recompute transcript features each run.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from features import build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)

def main():
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    t0 = time.time()
    X = build_features(f, os.path.join(ROOT, "data", "train_transcripts"))
    print(f"built features {X.shape} in {time.time()-t0:.1f}s")
    X.to_parquet(os.path.join(CACHE, "train_X.parquet"))
    print("saved ->", os.path.join(CACHE, "train_X.parquet"))

if __name__ == "__main__":
    main()
