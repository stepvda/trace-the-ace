"""Build a domain/task-adaptive MLM corpus = MathDial tutoring dialogues
(CC-BY-SA) + our own competition transcripts, both role-tagged like our task
input. One document per line -> cache/pretrain_corpus.txt
"""
import os, sys, re, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MATHDIAL = "/tmp/mathdial/data"


def mathdial_docs():
    docs = []
    path = os.path.join(MATHDIAL, "train.jsonl")
    if not os.path.exists(path):
        path = os.path.join(MATHDIAL, "train.csv")
    try:
        if path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path) if l.strip()]
        else:
            rows = pd.read_csv(path).to_dict("records")
    except Exception as e:
        print("mathdial load err", e); return docs
    for r in rows:
        conv = r.get("conversation") or ""
        if not conv:
            continue
        turns = str(conv).split("|EOM|")
        out = []
        for t in turns:
            t = t.strip()
            # normalise "Teacher: (move)text" -> "T: text", "<Name>: text" -> "S: text"
            t = re.sub(r"^Teacher:\s*(\([^)]*\))?", "T: ", t)
            t = re.sub(r"^[A-Z][a-z]+:\s*", "S: ", t)  # student name -> S:
            out.append(t)
        doc = " ".join(out).strip()
        if len(doc.split()) >= 10:
            docs.append(doc)
    return docs


def own_docs(max_docs=35000):
    p = os.path.join(ROOT, "submission", "assets", "train_texts.parquet")
    if not os.path.exists(p):
        return []
    df = pd.read_parquet(p)
    # strip the "Objective: ... Dialogue:" prefix -> keep the dialogue
    docs = []
    for t in df["text"].tolist()[:max_docs]:
        t = re.sub(r"^Objective:.*?Dialogue:\s*", "", str(t))
        if len(t.split()) >= 10:
            docs.append(t)
    return docs


def main():
    md = mathdial_docs()
    own = own_docs()
    print(f"MathDial docs={len(md)}  own transcript docs={len(own)}")
    all_docs = md + own
    out = os.path.join(CACHE, "pretrain_corpus.txt")
    with open(out, "w") as f:
        for d in all_docs:
            f.write(re.sub(r"\s+", " ", d).strip() + "\n")
    print(f"wrote {len(all_docs)} docs -> {out} "
          f"({os.path.getsize(out)/1e6:.1f} MB, ~{sum(len(d.split()) for d in all_docs)//1000}k words)")


if __name__ == "__main__":
    main()
