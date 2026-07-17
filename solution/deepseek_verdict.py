"""Definitive test of the LLM-as-extractor premise, using a FRONTIER model.

The local 3B extractor was a null, but a 3B cannot do the reasoning-quality judgment
the idea depends on. This asks DeepSeek (frontier-class, far beyond any bundleable
7-8B) for the SAME 3-way mastery verdict on the SAME 572 rows, prompt, and turn
selection as the local runs. Logic: if even a top model's verdict does NOT beat the
classical, no bundled model will -> the idea is dead. If it DOES, there is a real
ceiling worth a container build.

RUN THIS YOURSELF (it sends transcript snippets to the external DeepSeek API, which
Claude's auto-mode guard blocks):
    .venv/bin/python solution/deepseek_verdict.py
Then Claude evaluates it (LLM_MODEL=deepseek makes llm_extract's OUT point here):
    LLM_MODEL=deepseek .venv/bin/python solution/llm_extract.py eval
    LLM_MODEL=deepseek .venv/bin/python solution/llm_extract.py stack

Concurrent + resumable: appends to cache/llm_scores_deepseek.csv; safe to re-run.
"""
import os, sys, json, re, ssl, urllib.request, urllib.error, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from llm_extract import sample_rows, build_turns, PROMPT, VERDICTS, CACHE

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")   # strong + fast; "deepseek-reasoner" for max
OUT = os.path.join(CACHE, "llm_scores_deepseek.csv")
WORKERS = int(os.environ.get("DEEPSEEK_WORKERS", "8"))

raw = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "key.txt")).read().strip()
_m = re.search(r"(sk-[A-Za-z0-9_\-]+)", raw)
KEY = _m.group(1) if _m else raw.split("=")[-1].strip().strip('"')
try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()


def classify(prompt, retries=3):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 4,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for _ in range(retries):
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120, context=_CTX))
            t = r["choices"][0]["message"]["content"].upper()
            for k, v in VERDICTS.items():
                if k in t:
                    return v
            return np.nan
        except Exception:
            time.sleep(2)
    return np.nan


def main():
    s = sample_rows()  # seed=0 -> identical 572 rows as the local runs
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(line.split(",")[0])
    fh = open(OUT, "a")
    if "response_id" not in done:
        fh.write("response_id,llm_v,y\n")
    rows = [r for _, r in s.iterrows() if str(r.response_id) not in done]
    print(f"{len(rows)} to do with {MODEL} ({WORKERS} workers)", flush=True)

    def work(r):
        turns = build_turns(str(r.session_id))
        v = np.nan if turns is None else classify(
            PROMPT.format(obj=str(r.learning_objective)[:300], turns=turns))
        return str(r.response_id), v, int(r.is_correct)

    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rid, v, y in (f.result() for f in as_completed([ex.submit(work, r) for r in rows])):
            fh.write(f"{rid},{v},{y}\n"); fh.flush(); n += 1
            if n % 40 == 0:
                print(f"  {n}/{len(rows)} ({(time.time()-t0)/n:.2f}s/call)", flush=True)
    fh.close()
    print(f"done {n} in {time.time()-t0:.0f}s -> {OUT}", flush=True)
    print("Now: .venv/bin/python solution/llm_extract.py stack   (with LLM_MODEL=deepseek)", flush=True)


if __name__ == "__main__":
    main()
