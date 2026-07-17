"""Inference entrypoint — CLASSICAL model (reliable July-14 primary submission).

Reads the read-only ./data (test_features.csv, test_transcripts/,
submission_format.csv), computes features identically to training, loads the
pre-fit classical pipeline from ./assets, and writes ./submission.csv.

Runs fully offline. Logs only generic progress (no test-data specifics), per the
rules. Calibration: the shipped artifacts carry shrink_a=0.68 + shrink_center=0.685
(affine recenter onto the estimated test base rate, fit to the 3 LB anchors);
predict_pipeline applies p -> 0.685 + 0.68*(p - 0.7025) internally.

(The ambitious A100 transformer-ensemble entrypoint is submission/main_container.py.)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import joblib

from features import build_features
import model  # noqa: F401  (ensures classes importable for unpickling)
from model import predict_pipeline

DATA = os.path.join(HERE, "data")
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "submission.csv")


def main():
    print("Loading model artifacts...", flush=True)
    art = joblib.load(os.path.join(ASSETS, "artifacts.pkl"))

    test_features = pd.read_csv(os.path.join(DATA, "test_features.csv"))
    sub_format = pd.read_csv(os.path.join(DATA, "submission_format.csv"))
    transcripts_dir = os.path.join(DATA, "test_transcripts")

    print("Building features...", flush=True)
    X = build_features(test_features, transcripts_dir)

    print("Running inference...", flush=True)
    p, _, _ = predict_pipeline(art, X)

    preds = pd.DataFrame({"response_id": X.index, "probability": p})
    out = sub_format[["response_id"]].merge(preds, on="response_id", how="left")
    # unmatched rows (if build_features ever drops one) get the best constant = the
    # recentered test rate (shrink_center), NOT the train mean — same calibration target
    # as the modelled rows, so a dropped row isn't silently mis-calibrated to 0.7025.
    fill = float(art["cfg"].get("shrink_center", art.get("global_mean", 0.5)))
    out["probability"] = out["probability"].fillna(fill)
    out["probability"] = out["probability"].clip(1e-4, 1 - 1e-4)
    out.to_csv(OUT, index=False)
    print("Wrote submission.csv", flush=True)


if __name__ == "__main__":
    main()
