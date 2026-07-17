"""Container-side LLM-as-extractor (vLLM on the A100), DeepSeek idea #1 / MODEL_ARCHITECTURE #1.

Reads each session transcript + learning objective and emits an ordinal mastery
verdict (2=MASTERED, 1=PARTIAL, 0=CONFUSED). It mirrors the LOCALLY-VALIDATED
extractor (solution/llm_extract.py) EXACTLY — same prompt, same middle-window turn
selection — so the A100's larger instruct model applies the identical, validated
logic. Fully offline: loads a bundled instruct model from disk.

Usage in the container:
    from llm_verdict_vllm import extract_verdicts
    v_test  = extract_verdicts(test_features,  test_tdir,  MODEL_DIR, log)
    v_train = extract_verdicts(train_features, train_tdir, MODEL_DIR, log)
then fold the verdict one-hot into the classical feature matrix (refit) or blend an
empirical P(correct | verdict) predictor via the existing held-out ensemble weighter.

Robustness: any failure (missing weights, OOM, vLLM import error) is caught by the
caller so the submission falls back to the classical+DL ensemble. Never logs
test-data specifics — only generic progress.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from llm_extract import PROMPT, VERDICTS, build_turns  # parity with local validation


def _parse(text):
    t = (text or "").upper()
    for k, v in VERDICTS.items():
        if k in t:
            return v
    return np.nan


def extract_verdicts(features_df, transcripts_dir, model_dir, log=print,
                     max_model_len=2048, dtype="bfloat16"):
    """Return a pd.Series of ordinal verdicts indexed by response_id (nan if no
    usable transcript). One LLM call per SESSION (cached), reused across all of a
    session's responses — sessions, not responses, are the unit of transcript."""
    from vllm import LLM, SamplingParams

    feats = features_df[["response_id", "session_id", "learning_objective"]].copy()
    feats["session_id"] = feats["session_id"].astype(str)

    # de-duplicate by session: build one prompt per unique session
    sess = feats.drop_duplicates("session_id")
    prompts, sids = [], []
    for _, r in sess.iterrows():
        turns = build_turns(str(r.session_id), transcripts_dir)
        if turns is None:
            continue
        prompts.append(PROMPT.format(obj=str(r.learning_objective)[:300], turns=turns))
        sids.append(str(r.session_id))
    log(f"LLM extractor: {len(prompts)} session prompts")
    if not prompts:
        return pd.Series(np.nan, index=features_df["response_id"].values)

    llm = LLM(model=model_dir, dtype=dtype, max_model_len=max_model_len,
              gpu_memory_utilization=0.55, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    outs = llm.generate(prompts, sp)
    verdict_by_sid = {sids[i]: _parse(outs[i].outputs[0].text) for i in range(len(sids))}

    v = feats.set_index("response_id")["session_id"].map(verdict_by_sid)
    log(f"LLM extractor: verdicts for {v.notna().sum()}/{len(v)} responses")
    return v.reindex(features_df["response_id"].values)


def empirical_prob(train_verdict, train_y, verdict):
    """Map a verdict Series to P(correct|verdict) learned on train (a coarse but
    decorrelated predictor for the ensemble). Unknown/nan verdict -> global mean."""
    tv = np.asarray(train_verdict, float)
    ty = np.asarray(train_y, float)
    gm = float(np.nanmean(ty))
    table = {}
    for k in (0, 1, 2):
        mk = tv == k
        table[k] = float(ty[mk].mean()) if mk.sum() >= 30 else gm
    return np.array([table.get(int(x), gm) if x == x else gm for x in np.asarray(verdict, float)])
