"""Download the harvested literature (arXiv + open-access PDFs) to literature/papers2/."""
import os, re, json, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = ("/private/tmp/claude-502/-Users-nstephane-Dev-AI-Data-Science-training-trace-the-ace/"
       "648de05e-289a-4509-a6b1-01c3253db3d6/tasks/w24d903sw.output")
DEST = os.path.join(ROOT, "literature", "papers2")
os.makedirs(DEST, exist_ok=True)

data = json.load(open(OUT))
sources = data["result"]["sources"]
json.dump(sources, open(os.path.join(ROOT, "literature", "harvest_sources.json"), "w"), indent=1)
print(f"{len(sources)} sources")

SKIP_HOST = ("semanticscholar.org", "doi.org", "researchgate.net", "tandfonline.com",
             "sciencedirect.com", "springer.com", "dl.acm.org/doi/10", "jstor.org")


def to_pdf_url(u):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", u)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([a-z\-]+/[0-9]{7})", u)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return u


def slug(t, i):
    s = re.sub(r"[^a-z0-9]+", "_", t.lower())[:60].strip("_")
    return f"{i:03d}_{s}.pdf"


ok = fail = skip = 0
for i, s in enumerate(sources):
    u = s.get("url", "")
    if not u or any(h in u.lower() for h in SKIP_HOST):
        skip += 1; continue
    url = to_pdf_url(u)
    fn = os.path.join(DEST, slug(s.get("title", "paper"), i))
    if os.path.exists(fn):
        ok += 1; continue
    try:
        subprocess.run(["curl", "-sSL", "--max-time", "45", "-o", fn, url],
                       capture_output=True, timeout=60)
    except Exception:
        pass
    sz = os.path.getsize(fn) if os.path.exists(fn) else 0
    if sz > 20000 and open(fn, "rb").read(5).startswith(b"%PDF"):
        ok += 1
    else:
        if os.path.exists(fn): os.remove(fn)
        fail += 1
    if i % 25 == 0:
        print(f"  [{i}/{len(sources)}] ok={ok} fail={fail} skip={skip}", flush=True)
    time.sleep(0.3)

print(f"DOWNLOAD DONE: ok={ok} fail={fail} skip={skip} -> {DEST}")
