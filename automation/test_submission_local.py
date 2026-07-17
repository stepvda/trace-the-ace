"""Locally simulate the code-execution runtime to validate submission/main.py.

Builds a temp dir laid out exactly like the container:
    run/
      main.py, features.py, model.py, assets/artifacts.pkl   (copied from submission/)
      data/
        test_features.csv, submission_format.csv, test_transcripts/*.csv
Uses a slice of the TRAINING data as stand-in test data, runs main.py, and
validates submission.csv against the format + computes log loss vs known labels.
"""
import os, sys, shutil, subprocess, tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")


def main(n=800, seed=0):
    feats = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    labs = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv"))
    samp = feats.sample(n=n, random_state=seed).reset_index(drop=True)

    run = tempfile.mkdtemp(prefix="ttace_run_")
    data = os.path.join(run, "data")
    tdir = os.path.join(data, "test_transcripts")
    os.makedirs(tdir, exist_ok=True)

    # copy code + assets
    for fn in ["main.py"]:
        shutil.copy(os.path.join(SUB, fn), os.path.join(run, fn))
    for fn in ["features.py", "model.py", "dl_common.py", "dl_train.py"]:
        src = os.path.join(ROOT, "solution", fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(run, fn))
    os.makedirs(os.path.join(run, "assets"), exist_ok=True)
    # skip the large base_model here (BASE_MODEL_DIR env overrides it for local runs)
    for fn in ["artifacts.pkl", "train_texts.parquet", "classical_oof.parquet"]:
        src = os.path.join(SUB, "assets", fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(run, "assets", fn))
    _bm = os.path.join(SUB, "assets", "base_model")
    if "BASE_MODEL_DIR" not in os.environ and os.path.isdir(_bm):
        shutil.copytree(_bm, os.path.join(run, "assets", "base_model"))

    # write test_features.csv (same schema as train_features)
    samp.to_csv(os.path.join(data, "test_features.csv"), index=False)
    # submission_format.csv
    pd.DataFrame({"response_id": samp.response_id, "probability": 0.5}).to_csv(
        os.path.join(data, "submission_format.csv"), index=False)
    # transcripts for the needed sessions
    for sid in samp.session_id.astype(str).unique():
        src = os.path.join(ROOT, "data", "train_transcripts", f"{sid}.csv")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tdir, f"{sid}.csv"))

    # run main.py exactly as the container would (cwd = run dir)
    print("running main.py in", run)
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "main.py"], cwd=run, capture_output=True, text=True, env=env)
    print("--- stdout ---\n", r.stdout)
    print("--- stderr ---\n", r.stderr[-2000:])
    assert r.returncode == 0, f"main.py failed rc={r.returncode}"

    out_path = os.path.join(run, "submission.csv")
    assert os.path.exists(out_path), "submission.csv not written"
    out = pd.read_csv(out_path)
    fmt = pd.read_csv(os.path.join(data, "submission_format.csv"))
    assert list(out.columns) == ["response_id", "probability"], f"bad cols {list(out.columns)}"
    assert len(out) == len(fmt), f"row count {len(out)} != {len(fmt)}"
    assert set(out.response_id) == set(fmt.response_id), "response_id mismatch"
    assert out.probability.between(0, 1).all(), "probabilities out of [0,1]"
    assert out.probability.notna().all(), "NaN probabilities"

    # score vs known labels (this is train data, so optimistic, but validates plumbing)
    m = out.merge(labs, on="response_id")
    ll = log_loss(m.is_correct, np.clip(m.probability, 1e-6, 1 - 1e-6))
    auc = roc_auc_score(m.is_correct, m.probability)
    print(f"\nOK: submission.csv valid. rows={len(out)} "
          f"prob[min={out.probability.min():.3f} max={out.probability.max():.3f} mean={out.probability.mean():.3f}]")
    print(f"(on-train) log_loss={ll:.4f} auc={auc:.4f}")
    shutil.rmtree(run, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    main(n)
