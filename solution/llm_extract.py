"""LLM-as-extractor — local validation of DeepSeek idea #1 / MODEL_ARCHITECTURE #1.

An instruct LLM reads a tutoring transcript + the learning objective and outputs
P(student answers the NEXT same-objective quiz question correctly). This is a
*fixed external function* (no fitting on train) so its correlation with the label
on held-out rows IS its transferable signal — there is no leakage to worry about,
unlike objective-derived features. Locally we proxy the A100's 7-8B vLLM model
with a 3B model via Ollama (offline, Metal). If a 3B shows signal, the container's
bigger model shows more.

Metrics: (a) point-biserial corr + AUC of the raw LLM score; (b) incremental
objective-grouped HGB AUC over the 64 numeric features on the same rows.

Resumable: appends each score to cache/llm_scores.csv so progress survives a stop.
"""
import os, sys, json, glob, time, urllib.request, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TRANS = os.path.join(ROOT, "data", "train_transcripts")
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
# per-model output so results from different extractor models never overwrite
OUT = os.path.join(CACHE, "llm_scores_" + re.sub(r"[^a-z0-9]+", "-", MODEL.lower()) + ".csv")
N_TURNS = 14        # last N non-background turns (short = faster prompt processing on M1)
MAXCH = 140         # per-utterance char cap

# A 3B model won't emit a calibrated probability (it anchors to one bucket for every
# transcript), but it CAN commit to a 3-way verdict that varies with the content and
# is directionally aligned with the outcome. We store the ordinal verdict; the eval
# learns P(correct | verdict, ...) out-of-fold. The A100's larger model can output a
# finer score, but the verdict is the robust, small-model-safe signal to validate.
VERDICTS = {"MASTERED": 2, "PARTIAL": 1, "CONFUSED": 0}
# Prompt targets exactly what lexical praise/uncertainty features CANNOT see: whether
# the student REASONED to the answer themselves vs was told/led to it (the
# "correct-answer trap"), and confident-but-wrong reasoning. That reasoning-quality
# judgment is the only place an LLM extractor can beat the classical model.
PROMPT = """You are an expert math tutor judging whether a student TRULY understands an objective.
OBJECTIVE: {obj}
TRANSCRIPT ([S]=student [T]=tutor):
{turns}

Judge the student's REASONING, not politeness, and NOT merely whether a final answer
sounded right. Watch specifically for:
- a correct answer reached only because the tutor told or led them step-by-step (shallow), vs
- the student INDEPENDENTLY explaining why / doing the work themselves (real understanding), vs
- confident but WRONG reasoning or an unresolved misconception.

Classify, using ONLY evidence above, with exactly one word:
MASTERED  - student independently produced correct reasoning or explained why
PARTIAL   - reached it only with substantial tutor guidance, or minor errors / hesitation
CONFUSED  - guessed, wrong reasoning, a misconception, or the answer only appeared after being told
One word only: """


_PLEASANTRY = re.compile(r"^\W*(hi|hello|hey|bye|bye-bye|goodbye|okay|ok|yeah|yes|no|mm-?hmm|"
                         r"thank you|thanks|great|good|see you|take care|have a (great|good)|"
                         r"merry christmas|happy|welcome|sure|alright|all right)\b", re.I)


def _substantive(content):
    c = str(content).strip()
    if len(c) < 15:
        return False
    if _PLEASANTRY.match(c) and len(c) < 40:
        return False
    return True


def build_turns(sid, transcripts_dir=TRANS):
    """The mastery signal is in the MIDDLE of the session (math work), not the tail
    (goodbyes) or head (greetings). Drop intro/outro + pleasantries, keep the
    densest central window of real exchanges. Shared by the local validator and the
    A100 container extractor so both apply identical logic."""
    p = os.path.join(transcripts_dir, sid + ".csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d = d[d.role.isin(["student", "tutor"])].reset_index(drop=True)
    if len(d) == 0:
        return None
    # drop first 2 / last 3 turns (greeting / closing), then keep substantive turns
    if len(d) > 8:
        d = d.iloc[2:-3]
    d = d[d.content.map(_substantive)]
    if len(d) == 0:
        return None
    # central window: bias toward the middle where the diagnostic exchange lives
    if len(d) > N_TURNS:
        start = (len(d) - N_TURNS) // 2
        d = d.iloc[start:start + N_TURNS]
    tag = {"student": "[S]", "tutor": "[T]"}
    return "\n".join(f"{tag[r.role]} {str(r.content)[:MAXCH]}" for _, r in d.iterrows())


def ask(prompt, retries=2):
    """Return the ordinal verdict (2=MASTERED,1=PARTIAL,0=CONFUSED) or nan."""
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 6}}).encode()
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            t = r.get("response", "").upper()
            for k, v in VERDICTS.items():
                if k in t:
                    return v
        except Exception:
            time.sleep(1)
    return np.nan


def sample_rows(n_obj=26, cap=22, seed=0):
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    l = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f = f.join(l, on="response_id")
    vc = f.learning_objective_id.value_counts()
    elig = vc[vc >= cap].index.tolist()
    rng = np.random.RandomState(seed)
    chosen = rng.choice(elig, size=min(n_obj, len(elig)), replace=False)
    parts = [f[f.learning_objective_id == o].sample(n=cap, random_state=seed) for o in chosen]
    s = pd.concat(parts).reset_index(drop=True)
    print(f"sample: {len(s)} rows, {s.learning_objective_id.nunique()} objectives, "
          f"base rate {s.is_correct.mean():.3f}", flush=True)
    return s


def extract():
    s = sample_rows()
    done = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = dict(zip(prev.response_id.astype(str), prev.llm_v))
    fh = open(OUT, "a")
    if not done:
        fh.write("response_id,llm_v,y\n")
    t0 = time.time()
    n = 0
    for i, row in s.iterrows():
        rid = str(row.response_id)
        if rid in done and not np.isnan(done[rid]):
            continue
        turns = build_turns(str(row.session_id))
        p = np.nan if turns is None else ask(
            PROMPT.format(obj=str(row.learning_objective)[:300], turns=turns))
        fh.write(f"{rid},{p},{int(row.is_correct)}\n"); fh.flush()
        n += 1
        if n % 20 == 0:
            print(f"  {n} done ({(time.time()-t0)/n:.1f}s/call)", flush=True)
    fh.close()
    print(f"extracted {n} new in {time.time()-t0:.0f}s", flush=True)


def evaluate():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    from features import numeric_feature_columns
    sc = pd.read_csv(OUT).dropna(subset=["llm_v"]).drop_duplicates("response_id", keep="last")
    sc["response_id"] = sc.response_id.astype(str)
    y = sc.y.to_numpy(float); v = sc.llm_v.to_numpy(float)
    inv = {2: "MASTERED", 1: "PARTIAL", 0: "CONFUSED"}
    print(f"\n=== LLM verdict alone (n={len(sc)}) ===", flush=True)
    for k in (2, 1, 0):
        mk = v == k
        if mk.sum():
            print(f"  {inv[k]:9s} n={int(mk.sum()):4d}  P(correct)={y[mk].mean():.3f}")
    print(f"  ordinal corr(verdict, y) = {np.corrcoef(v, y)[0,1]:+.3f}")
    # out-of-fold empirical P(correct|verdict) -> honest AUC of the verdict signal
    oofp = np.zeros(len(y))
    from sklearn.model_selection import KFold
    for tr, va in KFold(5, shuffle=True, random_state=0).split(v):
        mp = {k: y[tr][v[tr] == k].mean() if (v[tr] == k).any() else y[tr].mean() for k in (0, 1, 2)}
        oofp[va] = [mp[int(x)] for x in v[va]]
    print(f"  OOF AUC(verdict) = {roc_auc_score(y, oofp):.4f}")

    # incremental over numeric features (same rows), objective-grouped
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    xset = set(X.index.astype(str))
    common = [r for r in sc.response_id if r in xset]
    Xi = X.loc[common]
    m = sc.set_index("response_id").loc[common]
    yy = m.y.to_numpy(float); vv = m.llm_v.to_numpy(int)
    oh = np.zeros((len(vv), 3)); oh[np.arange(len(vv)), vv] = 1.0  # one-hot verdict
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    groups = f.loc[common, "learning_objective_id"].astype(str).to_numpy()
    num = Xi[numeric_feature_columns(Xi)].to_numpy(float)
    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med)
    num = np.where(np.isnan(num), med, num)

    def cv(M):
        oof = np.zeros(len(yy))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(M, yy, groups):
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                    l2_regularization=1.0, min_samples_leaf=30, early_stopping=True, random_state=0)
            c.fit(M[tr], yy[tr]); oof[va] = c.predict_proba(M[va])[:, 1]
        return roc_auc_score(yy, oof), log_loss(yy, np.clip(oof, 1e-6, 1-1e-6))
    print(f"\n=== incremental over {num.shape[1]} numeric feats (n={len(common)}, obj-grouped) ===", flush=True)
    b_auc, b_ll = cv(num)
    w_auc, w_ll = cv(np.hstack([num, oh]))
    print(f"  numeric only         : auc={b_auc:.4f} ll={b_ll:.5f}")
    print(f"  numeric + llm_verdict: auc={w_auc:.4f} ll={w_ll:.5f}")
    print(f"  effect: {w_auc-b_auc:+.4f} AUC, {b_ll-w_ll:+.5f} logloss")
    print("  => HELPS" if w_auc > b_auc + 0.001 else "  => no clear incremental gain")


def stack_test():
    """The container-representative test: does the verdict improve on the CLASSICAL
    model (whose OOF already contains TF-IDF + all numeric features)? Runs the exact
    llm_stack decision on the extracted subset -> tells us if the in-container
    integration would actually fire."""
    import llm_stack as S
    sc = pd.read_csv(OUT).dropna(subset=["llm_v"]).drop_duplicates("response_id", keep="last")
    sc["response_id"] = sc.response_id.astype(str)
    oof = pd.read_parquet(os.path.join(ROOT, "submission", "assets", "classical_oof.parquet"))
    oof["response_id"] = oof.response_id.astype(str)
    m = sc.merge(oof[["response_id", "p_classical"]], on="response_id", how="inner")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    groups = f.loc[m.response_id, "learning_objective_id"].astype(str).to_numpy()
    print(f"\n=== container-representative: verdict on top of CLASSICAL (n={len(m)}) ===", flush=True)
    # in-sample decision (same rows as base + test, just to read the gain sign/size)
    _, info = S.evaluate_and_apply(m.p_classical.to_numpy(float), m.llm_v.to_numpy(float),
                                   m.y.to_numpy(float), groups,
                                   m.p_classical.to_numpy(float), m.llm_v.to_numpy(float))
    print("  ->", {k: (round(x, 5) if isinstance(x, float) else x) for k, x in info.items()})


if __name__ == "__main__":
    if "eval" in sys.argv:
        evaluate()
    elif "stack" in sys.argv:
        stack_test()
    else:
        extract()
        evaluate()
