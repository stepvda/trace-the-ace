"""INFERENCE-ONLY container entrypoint (the pre-train-and-bundle design).

No in-container training. Loads: (1) the bundled pre-fit classical model, (2) the bundled
pre-fine-tuned ModernBERT seed checkpoints (assets/mbert_seeds/<rep>_seed<seed>/), (3) the
blend config chosen offline on OOF (assets/blend_config.json: active_reps, blend_w, platt_A/B).

Flow: classical RAW probs + transformer ensemble probs -> blend (w) -> Platt calibration -> write.
The transformer is predicted with flash-attention if available, else a batch-size-1 SDPA fallback
(batch 1 has no padding, so it sidesteps the ModernBERT sdpa-on-padding NaN bug). Any failure in
the transformer path falls back to classical-only with the validated shrink/recenter calibration.
Never logs test specifics (rules).
"""
import os, sys, json, warnings, traceback
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd, joblib

from features import build_features
import model  # noqa: F401
from model import predict_pipeline
import dl_common

DATA = os.path.join(HERE, "data")
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "submission.csv")

# rep -> (max_len, dl_common builder knobs) — MUST match gpu_mbert.ARMS used for training
REP_CFG = {
    "control": dict(max_len=3072, HISTORY_WORDS=0,   RELEVANT_WORDS=600, RECENT_WORDS=1000),
    "history": dict(max_len=3072, HISTORY_WORDS=400, RELEVANT_WORDS=600, RECENT_WORDS=1000),
}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _build_rep_texts(rep, features_df, tdir):
    c = REP_CFG[rep]
    dl_common.HISTORY_WORDS = c["HISTORY_WORDS"]
    dl_common.RELEVANT_WORDS = c["RELEVANT_WORDS"]
    dl_common.RECENT_WORDS = c["RECENT_WORDS"]
    return dl_common.build_texts(features_df, tdir, n_words=1600, centered=True, proxy_tags=True), c["max_len"]


def _predict_one(model_dir, texts, max_len, log):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir); tok.truncation_side = "left"
    batched = True
    try:  # flash-attn: batched inference (bf16)
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, reference_compile=False, attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16).to(device).eval()
    except Exception:  # no flash-attn -> sdpa with batch-1 (no padding -> no NaN)
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, reference_compile=False, attn_implementation="sdpa").to(device).eval()
        batched = False
        log("[dl] flash-attn unavailable -> SDPA batch-1 fallback")
    out = []
    with torch.no_grad():
        if batched:
            for i in range(0, len(texts), 32):
                enc = tok(list(texts[i:i + 32]), truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to(device)
                with torch.autocast(device, dtype=torch.bfloat16):
                    out.append(torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        else:
            for t in texts:  # batch size 1, no padding
                enc = tok([t], truncation=True, max_length=max_len, padding=False, return_tensors="pt").to(device)
                out.append(torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    print("Loading classical artifacts...", flush=True)
    art = joblib.load(os.path.join(ASSETS, "artifacts.pkl"))
    # classical leg must be RAW (unshrunk) to match classical_oof / the blend fit
    cal_a = float(art["cfg"].get("shrink_a", 1.0))
    cal_center = float(art["cfg"].get("shrink_center", art["global_mean"]))
    try:
        cal_prior = float(pd.read_parquet(os.path.join(ASSETS, "classical_oof.parquet"))["p_classical"].mean())
    except Exception:
        cal_prior = float(art["global_mean"])
    art["cfg"] = dict(art["cfg"], shrink_a=1.0); art["cfg"].pop("shrink_center", None)

    test_features = pd.read_csv(os.path.join(DATA, "test_features.csv"))
    sub_format = pd.read_csv(os.path.join(DATA, "submission_format.csv"))
    tdir = os.path.join(DATA, "test_transcripts")

    print("Classical features + RAW predictions...", flush=True)
    Xc = build_features(test_features, tdir)
    p_classical, _, _ = predict_pipeline(art, Xc)
    p_classical = pd.Series(p_classical, index=Xc.index)

    final = None
    try:
        cfg = json.load(open(os.path.join(ASSETS, "blend_config.json")))
        active_reps = cfg["active_reps"]; w = float(cfg["blend_w"])
        A = float(cfg["platt_A"]); B = float(cfg["platt_B"])
        print(f"Blend config: reps={active_reps} w={w} plattA={A} plattB={B}", flush=True)
        seeds_root = os.path.join(ASSETS, "mbert_seeds")
        preds = []
        for rep in active_reps:
            texts, max_len = _build_rep_texts(rep, test_features, tdir)
            for d in sorted(os.listdir(seeds_root)):
                if not d.startswith(rep + "_seed"):
                    continue
                mdir = os.path.join(seeds_root, d)
                p = _predict_one(mdir, texts, max_len, log=lambda m: print(m, flush=True))
                preds.append(p)
                print(f"[dl] {d} predicted (mean={p.mean():.4f})", flush=True)
        if not preds:
            raise RuntimeError("no seed checkpoints found")
        p_tr = pd.Series(np.mean(preds, axis=0), index=test_features["response_id"].values)
        p_tr = p_tr.reindex(Xc.index)
        blend = w * p_tr.values + (1 - w) * p_classical.values          # blend RAW legs
        final = pd.Series(_sigmoid(A * _logit(blend) + B), index=Xc.index)  # Platt calibration
        print("Transformer ensemble blended + calibrated.", flush=True)
    except Exception:
        print("DL/blend step failed -> classical-only fallback with shrink/recenter.", flush=True)
        traceback.print_exc()
        final = pd.Series(cal_center + cal_a * (p_classical.values - cal_prior), index=Xc.index)

    out = sub_format[["response_id"]].merge(
        pd.DataFrame({"response_id": final.index, "probability": final.values}),
        on="response_id", how="left")
    out["probability"] = out["probability"].fillna(cal_center).clip(1e-4, 1 - 1e-4)
    out.to_csv(OUT, index=False)
    print("Wrote submission.csv", flush=True)


if __name__ == "__main__":
    main()
