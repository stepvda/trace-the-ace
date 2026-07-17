"""Consult DeepSeek-reasoner about the Trace the Ace project.
Manifest-first flow: (1) give overview + list available materials, ask what it
wants to study; (2) send exactly what it requests, then ask for ranked ideas.
Saves the result to deepseek_ideas.md.

RUN THIS YOURSELF (it calls the external DeepSeek API, which Claude's auto-mode
guard blocks):  .venv/bin/python solution/ask_deepseek.py
"""
import json, urllib.request, urllib.error, re, os, glob, ssl
import pandas as pd, numpy as np

raw = open("key.txt").read().strip()
m = re.search(r"(sk-[A-Za-z0-9_\-]+)", raw)
KEY = m.group(1) if m else raw.split("=")[-1].strip().strip('"')

# macOS python.org builds often lack a CA bundle -> use certifi's
try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()


def call(messages, mt=8000):
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({"model": "deepseek-reasoner", "messages": messages,
                         "max_tokens": mt, "stream": False}).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900, context=_CTX))
    except urllib.error.HTTPError as e:
        print("HTTP ERROR", e.code, e.read().decode()[:800], flush=True)
        raise
    return r["choices"][0]["message"]["content"]


def data_context():
    f = pd.read_csv("data/train_features.csv"); l = pd.read_csv("data/train_labels.csv")
    tf = sorted(glob.glob("data/train_transcripts/*.csv"))
    nr = [len(pd.read_csv(x, dtype=str)) for x in tf[:60]]
    return (f"train_features cols={list(f.columns)} rows={len(f)}; labels mean={l.is_correct.mean():.4f}; "
            f"n_sessions={f.session_id.nunique()} n_objectives={f.learning_objective_id.nunique()}; "
            f"resp/session mean={f.groupby('session_id').size().mean():.2f} max={f.groupby('session_id').size().max()}; "
            f"transcript cols=[session_id,utterance_id,role(tutor/student/background),content,timestamp(elapsed HH:MM:SS)]; "
            f"utterances/session mean~{int(np.mean(nr))}")


def transcripts(n):
    f = pd.read_csv("data/train_features.csv"); l = pd.read_csv("data/train_labels.csv")
    tf = sorted(glob.glob("data/train_transcripts/*.csv"))[:n]
    out = []
    for x in tf:
        sid = os.path.basename(x)[:-4]
        d = pd.read_csv(x, dtype=str, keep_default_na=False)
        lab = l[l.response_id.isin(f[f.session_id == sid].response_id)]
        out.append(f"\n--- session {sid} label={lab.is_correct.tolist()} (first 30 turns) ---")
        for _, r in d.head(30).iterrows():
            out.append(f"[{r['timestamp']}] {r['role']}: {r['content'][:160]}")
    return "\n".join(out)


ITEMS = {
    1: ("README.md", lambda: open("README.md").read()),
    2: ("docs/SOLUTION.md (full methodology, features, models, data schema)", lambda: open("docs/SOLUTION.md").read()),
    3: ("docs/EXPERIMENT_LOG.md (every measure tried, HELPED/HURT with numbers)", lambda: open("docs/EXPERIMENT_LOG.md").read()),
    4: ("docs/MODEL_ARCHITECTURE.md (shallow vs sequence/semantic analysis)", lambda: open("docs/MODEL_ARCHITECTURE.md").read()),
    5: ("docs/RESULTS_AND_STRATEGY.md (leaderboard journey, calibration lessons)", lambda: open("docs/RESULTS_AND_STRATEGY.md").read()),
    6: ("literature/INSIGHTS.md (33-idea catalog synthesized from 290 papers)", lambda: open("literature/INSIGHTS.md").read()),
    7: ("solution/features.py (actual feature-engineering code)", lambda: open("solution/features.py").read()),
    8: ("data schema + summary statistics", data_context),
    9: ("sample transcripts (say how many; 35,072 exist, can paste up to ~30)", lambda: transcripts(10)),
    10: ("literature/harvest_sources.json (the 290 harvested sources)", lambda: open("literature/harvest_sources.json").read()[:20000]),
}
manifest = "\n".join(f"  [{k}] {v[0]}" for k, v in ITEMS.items())

SYS = {"role": "system", "content": "You are a world-class Kaggle grandmaster and ML researcher. The user faces a near-noise educational-outcome prediction task (predict if a student answers the NEXT quiz question correctly from a tutoring-session transcript; metric=log loss). They want ideas that will move a held-out objective-grouped metric AND transfer under train->test distribution shift. Be specific, quantitative, novel, brutally honest."}

overview = ("PROJECT: DrivenData 'Trace the Ace'. Predict per-response binary is_correct (student answers next same-topic "
            "quiz question right) from a student-tutor transcript + learning-objective text. Metric: LOG LOSS (AUROC secondary). "
            "35,072 train responses / 22,821 sessions / 398 objectives; base rate 0.70; constant baseline logloss 0.6088; leaderboard #1=0.6013 (near-noise). "
            "Code-execution submission: offline container, Python 3.12, A100 80GB + vLLM, 6h, 3 subs/week. Local dev machine is an 8GB M1 (no CUDA). "
            "Current best submission 0.6144 (#27/331). Honest metric = OBJECTIVE-GROUPED CV (it overestimates the leaderboard by ~0.04 AUC due to a TSL/Eedi provider shift). "
            "Briefly validated (objective-grouped): talk-move features +0.0102 AUC, MathDial domain-adaptive pretraining +0.0088, classical+transformer ensemble +0.008, calibration(unshrink) is the biggest lever. "
            "Hurt: learning-objective target-encoding (leakage), over-shrinkage, objective-derived difficulty/coverage feats, a LEXICAL in-session-correctness proxy.")

m1 = {"role": "user", "content": overview + "\n\nI can share any of these materials with you:\n" + manifest +
      "\n\nBefore giving ideas: tell me EXACTLY which items you want to study (list the numbers), and if you want #9, how many transcripts. "
      "Request anything else you'd find useful. I'll send them next. Do NOT give ideas yet."}
print("... turn 1: manifest -> asking DeepSeek what it wants ...", flush=True)
r1 = call([SYS, m1])
print("=== DEEPSEEK REQUESTS ===\n" + r1 + "\n", flush=True)

nums = sorted(set(int(x) for x in re.findall(r"\b(?:\[)?(\d{1,2})\b", r1) if 1 <= int(x) <= 10))
if not nums:
    nums = [2, 3, 4, 5, 6, 7, 8]
tm = re.search(r"(\d+)\s*(?:sample\s*)?transcript", r1.lower())
ntrans = min(int(tm.group(1)), 30) if tm else 10
print(f"[parsed -> items {nums}, transcripts {ntrans}]", flush=True)

payload = []
for k in nums:
    if k == 9:
        payload.append(f"\n\n===== [9] {ntrans} SAMPLE TRANSCRIPTS =====\n" + transcripts(ntrans))
    elif k in ITEMS:
        payload.append(f"\n\n===== [{k}] {ITEMS[k][0]} =====\n" + ITEMS[k][1]())
for k in (3, 4):
    if k not in nums:
        payload.append(f"\n\n===== [{k}] {ITEMS[k][0]} =====\n" + ITEMS[k][1]())
pay = "\n".join(payload)[:180000]

m2 = {"role": "user", "content": "Here is what you requested:\n" + pay +
      "\n\n---\nNow give your BEST novel, specific, actionable ideas to improve the model. RANK them. For each: (a) concrete how-to, "
      "(b) why it should help TRANSFER on this shifted near-noise task (not just fit CV), (c) feasibility (local 8GB M1 vs A100/vLLM container), "
      "(d) honest expected magnitude. Prioritize what's feasible in the A100/vLLM offline container. Include >=1 idea absent from my writeup, and >=1 way to "
      "better ESTIMATE/CALIBRATE the leaderboard from local data. Be concrete enough to implement."}
print("... turn 2: sending requested materials -> asking for ideas ...", flush=True)
r2 = call([SYS, m1, {"role": "assistant", "content": r1}, m2], mt=8000)
print("=== DEEPSEEK IDEAS ===\n" + r2, flush=True)
open("deepseek_ideas.md", "w").write("# DeepSeek — what it requested\n\n" + r1 +
                                      f"\n\n(sent items {nums}, {ntrans} transcripts)\n\n# DeepSeek — ideas\n\n" + r2)
print("\n[saved -> deepseek_ideas.md]", flush=True)
