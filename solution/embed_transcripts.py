"""Extract frozen sentence-embedding features from transcripts (PoC for whether
semantic embeddings add signal over TF-IDF).

Builds two recency-focused views per session and embeds them with a small
encoder (forward pass only -> low memory, fits 8GB M1):
  - view 'recent': last ~N words of the whole dialogue (role-tagged)
  - view 'student': last ~N words of student-only utterances
Saves arrays aligned to response_id under solution/cache/emb_*.npy
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def recency_texts(features_df, transcripts_dir, n_words=350):
    """For each session build (recent_all, recent_student) recency text."""
    out = {}
    for sid in features_df["session_id"].astype(str).unique():
        p = os.path.join(transcripts_dir, f"{sid}.csv")
        if not os.path.exists(p):
            out[sid] = ("", ""); continue
        try:
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:
            out[sid] = ("", ""); continue
        role = d.get("role", pd.Series([""] * len(d))).astype(str).str.lower()
        content = d.get("content", pd.Series([""] * len(d))).astype(str)
        # role-tagged lines
        lines = [f"{r}: {c}" for r, c in zip(role, content)]
        all_txt = " ".join(lines)
        stud_txt = " ".join(content[role.eq("student")].tolist())
        # keep last n_words words (recency, near the assessment)
        aw = all_txt.split(); sw = stud_txt.split()
        out[sid] = (" ".join(aw[-n_words:]), " ".join(sw[-n_words:]))
    return out


def main(model_name="BAAI/bge-small-en-v1.5", split="train", n_words=350, batch=64):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev} model={model_name}", flush=True)
    feats = pd.read_csv(os.path.join(ROOT, "data", f"{split}_features.csv"))
    tdir = os.path.join(ROOT, "data", f"{split}_transcripts")
    t0 = time.time()
    rec = recency_texts(feats, tdir, n_words=n_words)
    print(f"built recency texts for {len(rec)} sessions in {time.time()-t0:.1f}s", flush=True)

    model = SentenceTransformer(model_name, device=dev)
    ids = feats["response_id"].tolist()
    all_texts = [rec[str(s)][0] for s in feats["session_id"]]
    stud_texts = [rec[str(s)][1] for s in feats["session_id"]]

    t0 = time.time()
    emb_all = model.encode(all_texts, batch_size=batch, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    print(f"encoded 'recent' {emb_all.shape} in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    emb_stud = model.encode(stud_texts, batch_size=batch, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)
    print(f"encoded 'student' {emb_stud.shape} in {time.time()-t0:.1f}s", flush=True)

    tag = model_name.split("/")[-1]
    np.save(os.path.join(CACHE, f"emb_{split}_recent_{tag}.npy"), emb_all.astype(np.float32))
    np.save(os.path.join(CACHE, f"emb_{split}_student_{tag}.npy"), emb_stud.astype(np.float32))
    pd.Series(ids).to_csv(os.path.join(CACHE, f"emb_{split}_ids.csv"), index=False)
    print("saved embeddings ->", CACHE, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_words", type=int, default=350)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    main(a.model, a.split, a.n_words, a.batch)
