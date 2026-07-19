"""INFERENCE-ONLY container entrypoint (pre-train-and-bundle design). No in-container training.

Loads (1) the bundled pre-fit classical model, (2) bundled pre-fine-tuned ModernBERT seed
checkpoints (assets/mbert_seeds/control_seed*/), (3) the offline-fit blend + calibration
(assets/blend_config.json). Writes submission.csv.

Calibration (Fable-validated, composes two DIFFERENT maps — do NOT double-apply):
    blend_raw = w*p_transformer_raw + (1-w)*p_classical_RAW           # classical leg is PRE-recenter
    p_platt   = sigmoid(A*logit(blend_raw) + B)                        # calibrate on TRAIN dist
    p_ship    = center + a*(p_platt - blend_pivot)                     # recenter TRAIN->TEST (LB-fit)
The classical-ONLY fallback keeps its own recenter (center + a*(p_classical_raw - classical_oof_mean)),
reproducing the known-0.6087 pipeline exactly.

Robustness: flash-attn batched inference, else batch-size-1 SDPA (no padding => no ModernBERT NaN);
per-seed time-budget degradation; finiteness guards; classical-only last resort. Logs the executed
path so the LB read is a signal read, not a plumbing read.
"""
import os, sys, json, time, warnings, traceback
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
TIME_BUDGET_S = float(os.environ.get("INFER_BUDGET_S", 5.4 * 3600))  # margin under the 6h wall

# rep -> (max_len, dl_common knobs) — MUST match gpu_mbert.ARMS["control"] used for training
REP_CFG = {"control": dict(max_len=3072, HISTORY_WORDS=0, RELEVANT_WORDS=600, RECENT_WORDS=1000)}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _build_rep_texts(rep, features_df, tdir):
    c = REP_CFG[rep]
    dl_common.HISTORY_WORDS = c["HISTORY_WORDS"]
    dl_common.RELEVANT_WORDS = c["RELEVANT_WORDS"]
    dl_common.RECENT_WORDS = c["RECENT_WORDS"]
    dl_common.SEG_MODE = "last"   # parity with training (validated selector)
    return dl_common.build_texts(features_df, tdir, n_words=1600, centered=True, proxy_tags=True), c["max_len"]


def _load_seed(model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("FORCE_SDPA") != "1":     # test hook: FORCE_SDPA=1 exercises the container path
        try:  # flash-attn: batched, bf16
            m = AutoModelForSequenceClassification.from_pretrained(
                model_dir, reference_compile=False, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16).to(device).eval()
            return m, device, "flash"
        except Exception:
            pass
    if True:  # no flash-attn (or forced) -> SDPA batch-1 (no padding -> no NaN), bf16 for speed
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, reference_compile=False, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16).to(device).eval()
        return m, device, "sdpa-b1"


def _predict(m, device, mode, tok, texts, max_len):
    import torch
    out = []
    with torch.no_grad():
        if mode == "flash":
            for i in range(0, len(texts), 32):
                enc = tok(list(texts[i:i + 32]), truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to(device)
                with torch.autocast(device, dtype=torch.bfloat16):
                    lg = m(**enc).logits.float()
                if not torch.isfinite(lg).all():
                    raise RuntimeError("non-finite logits (flash)")
                out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        else:  # sdpa batch-1
            for t in texts:
                enc = tok([t], truncation=True, max_length=max_len, padding=False, return_tensors="pt").to(device)
                lg = m(**enc).logits.float()
                if not torch.isfinite(lg).all():
                    raise RuntimeError("non-finite logits (sdpa-b1)")
                out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    print("Loading classical artifacts...", flush=True)
    art = joblib.load(os.path.join(ASSETS, "artifacts.pkl"))
    cal_a = float(art["cfg"].get("shrink_a", 1.0))
    cal_center = float(art["cfg"].get("shrink_center", art["global_mean"]))
    try:
        fallback_prior = float(pd.read_parquet(os.path.join(ASSETS, "classical_oof.parquet"))["p_classical"].mean())
    except Exception:
        fallback_prior = float(art["global_mean"])
    art["cfg"] = dict(art["cfg"], shrink_a=1.0); art["cfg"].pop("shrink_center", None)  # classical -> RAW

    test_features = pd.read_csv(os.path.join(DATA, "test_features.csv"))
    sub_format = pd.read_csv(os.path.join(DATA, "submission_format.csv"))
    tdir = os.path.join(DATA, "test_transcripts")

    print("Classical features + RAW predictions...", flush=True)
    Xc = build_features(test_features, tdir)
    p_classical = pd.Series(predict_pipeline(art, Xc)[0], index=Xc.index)

    leg, used_attn, seeds_used = "classical-fallback", "none", 0
    final = None
    try:
        import torch
        from transformers import AutoTokenizer
        bc = json.load(open(os.path.join(ASSETS, "blend_config.json")))
        rep = bc["active_reps"][0]                    # control
        w = float(bc["blend_w"]); A = float(bc["platt_A"]); B = float(bc["platt_B"])
        pivot = float(bc["blend_pivot"]); center = float(bc["recenter_center"]); a = float(bc["recenter_a"])
        seeds_root = os.path.join(ASSETS, "mbert_seeds")
        seed_dirs = sorted(d for d in os.listdir(seeds_root) if d.startswith(rep + "_seed"))
        texts, max_len = _build_rep_texts(rep, test_features, tdir)
        tok = AutoTokenizer.from_pretrained(os.path.join(seeds_root, seed_dirs[0])); tok.truncation_side = "left"

        preds = []
        for i, d in enumerate(seed_dirs):
            m, device, mode = _load_seed(os.path.join(seeds_root, d)); used_attn = mode
            # coarse budget pre-check on seed 0 (batch-1 can be slow): project full-seed time
            if i == 0:
                n_probe = min(200, len(texts)); ts = time.time()
                _ = _predict(m, device, mode, tok, texts[:n_probe], max_len)
                est_per_seed = (time.time() - ts) / max(1, n_probe) * len(texts)
                print(f"[dl] attn={mode} est_per_seed={est_per_seed:.0f}s n_test={len(texts)}", flush=True)
                if time.time() - t0 + est_per_seed > TIME_BUDGET_S:
                    print("[dl] a single seed would exceed the time budget -> classical-only", flush=True)
                    del m; raise RuntimeError("time budget")
            ts = time.time()
            p = _predict(m, device, mode, tok, texts, max_len)
            if not np.isfinite(p).all():
                raise RuntimeError("non-finite seed predictions")
            preds.append(p); per_seed = time.time() - ts
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[dl] {d} done ({per_seed:.0f}s, mean={p.mean():.4f})", flush=True)
            if time.time() - t0 + per_seed > TIME_BUDGET_S:  # no budget for another seed
                print(f"[dl] stopping at {len(preds)} seeds (time budget)", flush=True)
                break
        if not preds:
            raise RuntimeError("no seed predictions")
        seeds_used = len(preds)
        p_tr = pd.Series(np.mean(preds, axis=0), index=test_features["response_id"].values).reindex(Xc.index)

        blend_raw = w * p_tr.values + (1 - w) * p_classical.values        # RAW legs
        p_platt = _sigmoid(A * _logit(blend_raw) + B)                     # calibrate-on-train
        final = pd.Series(center + a * (p_platt - pivot), index=Xc.index)  # recenter train->test
        if not np.isfinite(final.values).all():
            raise RuntimeError("non-finite final blend")
        leg = "blend"
    except Exception:
        print("DL/blend step failed or budget-gated -> classical-only fallback.", flush=True)
        traceback.print_exc()
        final = pd.Series(cal_center + cal_a * (p_classical.values - fallback_prior), index=Xc.index)
        leg = "classical-fallback"

    print(f"PATH attn={used_attn} seeds_used={seeds_used} leg={leg} elapsed={int(time.time()-t0)}s", flush=True)
    out = sub_format[["response_id"]].merge(
        pd.DataFrame({"response_id": final.index, "probability": final.values}),
        on="response_id", how="left")
    out["probability"] = out["probability"].fillna(cal_center).clip(1e-4, 1 - 1e-4)
    out.to_csv(OUT, index=False)
    print("Wrote submission.csv", flush=True)


if __name__ == "__main__":
    main()
