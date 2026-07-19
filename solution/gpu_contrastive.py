"""Round-6 idea #6: contrastive objective<->segment MATCHING pretrain (task-shaped inductive bias
instead of generic MLM). Warm ModernBERT-base on a self-supervised matching task that mirrors the
downstream CROSS-encoder, then fine-tune for correctness from that init and measure the OOF.

Design (revised per Fable's pre-spend design review):
- Positive: the row's REAL centered text  "Objective: <lo>. Relevant: <content about lo> Recent: <tail>".
- Negative: the SAME transcript body, only the leading "Objective: <lo>." swapped to a DIFFERENT
  objective (80% HARD: pooled 200, ranked by overlap RATIO, near-duplicates Jaccard>0.6 EXCLUDED
  because near-synonym objectives genuinely match the body -> false negatives; else random).
- SHARED-TERM DROPOUT: before pairing, delete surface occurrences of a random ~50% subset of the true
  objective's content-terms from the body (applied ONCE -> +/- bodies stay byte-identical). Without
  this the matching task is trivial COPY-DETECTION (the Relevant-selector only emits a segment when it
  already shares >=2 objective terms), a skill the encoder + representation already have -> no transfer.
  Dropout forces SEMANTIC alignment (the actual bias we want to inject).
- Only rows that have a "Relevant:" segment are used (~70%); Dialogue-fallback rows dropped.
- truncation_side='right' here (Objective+Relevant sit at the START and must survive); the downstream
  OOF keeps its usual left-truncation.
- 5% of objectives HELD OUT of pretraining; match-acc logged on them each epoch/periodically. If held-out
  acc saturates fast/high -> task is trivial (copy-detection generalizes) -> expect no transfer; if it
  lags -> genuine semantic skill. Converts a downstream null into a diagnosis.
- No is_correct labels touched -> zero label leakage. Pretrain sees objective TEXT self-supervised;
  optimistic vs the honest per-fold regime, so a POSITIVE OOF must be reconfirmed per-fold before ship.

ModernBERT-base num_labels=2, flash_attention_2, fp32 master weights + bf16 autocast (the no-NaN recipe).
Saves to /workspace/mbert_contrastive/. Then: MBERT_DIR=<that> RESET_HEAD=1 python gpu_oof.py control ...
Usage: python gpu_contrastive.py <epochs=1> <max_len=1536> [subset]   (UNSET MBERT_DIR first!)
"""
import os, sys, time, math, random, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import dl_common as D
from gpu_mbert import load_model, ROOT, BASE, TDIR

OUT_DIR = os.environ.get("CONTRASTIVE_OUT", "/workspace/mbert_contrastive")
_WORDRE = re.compile(r"[A-Za-z']+")   # MATCH features.WORD_RE (incl apostrophes) so no term escapes dropout
_PROTECT = {"relevant", "recent", "history", "dialogue", "objective"}   # structural field labels
BODY_WORDS = 1000    # cap the body tail so Objective+body fit under max_len 1536 -> +/- stay
#                      byte-identical AFTER tokenization (right-trunc otherwise clips unequal tails)


def _content_terms(s):
    return D._content_terms(s)


def _drop_terms(body, obj_terms, rng, drop_frac=0.5):
    """Remove surface occurrences of a random ~drop_frac subset of the objective's content-terms from
    the body (kills copy-detection; forces semantic matching). Applied ONCE per row, before +/- pairing,
    so the two bodies stay byte-identical. Never drops structural field labels."""
    droppable = sorted(t for t in obj_terms if t not in _PROTECT)
    if not droppable:
        return body
    k = max(1, int(round(len(droppable) * drop_frac)))
    drop = set(rng.sample(droppable, min(k, len(droppable))))
    return _WORDRE.sub(lambda m: "" if m.group(0).lower() in drop else m.group(0), body)


def _neg_objective(lo_i, pool, terms, rng, hard_frac=0.8):
    """Pick a negative objective from `pool` (which EXCLUDES held-out objectives, so they never appear
    even as training negatives). Hard = high overlap RATIO but NOT a near-duplicate/subset (those
    genuinely match the body -> false negatives)."""
    ti = terms[lo_i]
    if rng.random() < hard_frac:
        cand = rng.sample(pool, min(200, len(pool)))
        scored = []
        for c in cand:
            if c == lo_i:
                continue
            tc = terms[c]; inter = len(tc & ti)
            if inter < 2:
                continue
            if inter / (len(tc | ti) or 1) > 0.6:              # near-duplicate -> false negative
                continue
            if inter / (min(len(ti), len(tc)) or 1) > 0.8:      # subset/containment -> false negative
                continue
            scored.append((inter / (len(tc) or 1), c))          # overlap RATIO wrt candidate terms
        if scored:
            return max(scored)[1]
    return rng.choice([u for u in pool if u != lo_i])


def build_pairs(f, seed, hard_frac=0.8, drop_frac=0.5, heldout_frac=0.05):
    """Return (texts, labels, heldout_mask, n_usable_rows). texts alternate positive, negative."""
    rng = random.Random(seed)
    D.HISTORY_WORDS = 0; D.RELEVANT_WORDS = 600; D.RECENT_WORDS = 1000
    turns_cache = {}
    rows = []                                   # (lo_str, body_raw)  body starts with " Relevant: ..."
    for lo, sid in zip(f["learning_objective"], f["session_id"].astype(str)):
        if sid not in turns_cache:
            p = os.path.join(TDIR, f"{sid}.csv"); df = None
            if os.path.exists(p):
                try: df = pd.read_csv(p, dtype=str, keep_default_na=False)
                except Exception: df = None
            turns_cache[sid] = D._parse_turns(df)
        txt = D.build_text_for_row(turns_cache[sid], lo, centered=True, proxy_tags=True)
        prefix = f"Objective: {'' if lo is None else str(lo)}."
        if "Relevant:" not in txt or not txt.startswith(prefix):
            continue                            # Dialogue-fallback / edge row: no matching signal
        rows.append((str(lo), txt[len(prefix):]))

    if not rows:
        raise ValueError("build_pairs: 0 usable rows (no 'Relevant:' segments) — check transcripts/paths")
    uniq = sorted({lo for lo, _ in rows})
    terms = {lo: _content_terms(lo) for lo in uniq}
    n_ho = max(1, int(round(len(uniq) * heldout_frac))) if heldout_frac > 0 else 0
    heldout_obj = set(rng.sample(uniq, n_ho)) if n_ho else set()
    neg_pool = [u for u in uniq if u not in heldout_obj]   # held-out objectives never seen, even as negatives

    texts, labels, held = [], [], []
    for lo_i, body_raw in rows:
        capped = " " + " ".join(body_raw.split()[:BODY_WORDS])       # bound tail (see BODY_WORDS)
        body = _drop_terms(capped, terms[lo_i], rng, drop_frac)      # once -> shared by +/-
        lo_j = _neg_objective(lo_i, neg_pool, terms, rng, hard_frac)
        h = lo_i in heldout_obj
        texts.append(f"Objective: {lo_i}.{body}"); labels.append(1); held.append(h)
        texts.append(f"Objective: {lo_j}.{body}"); labels.append(0); held.append(h)
    return texts, np.array(labels, dtype=np.int64), np.array(held, dtype=bool), len(rows)


class DS(Dataset):
    def __init__(self, texts, ys): self.texts = texts; self.ys = ys
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], int(self.ys[i])


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
    subset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    device = "cuda"
    log = lambda m: print(m, flush=True)
    log(f"=== CONTRASTIVE matching pretrain epochs={epochs} max_len={max_len} base={BASE} "
        f"gpu={torch.cuda.get_device_name(0)} ===")
    if BASE != "/workspace/models/ModernBERT-base" and not subset:
        log(f"!! WARNING: BASE={BASE} (stale MBERT_DIR?) — pretrain should start from STOCK ModernBERT-base")
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    if subset:
        f = f.sample(n=subset, random_state=0).reset_index(drop=True)
    t0 = time.time()
    texts, y, held, n_usable = build_pairs(f, seed=0)
    tr_idx = np.where(~held)[0]; ho_idx = np.where(held)[0]
    tr_texts = [texts[i] for i in tr_idx]; tr_y = y[tr_idx]
    ho_texts = [texts[i] for i in ho_idx]; ho_y = y[ho_idx]
    log(f"pairs={len(texts)} from {n_usable}/{len(f)} usable rows | train={len(tr_idx)} "
        f"heldout={len(ho_idx)} (objectives held out for match-acc diagnostic) ({int(time.time()-t0)}s)")

    tok = AutoTokenizer.from_pretrained(BASE); tok.truncation_side = "right"   # keep Objective+Relevant

    def collate(items):
        txt = [a for a, _ in items]; ys = [b for _, b in items]
        enc = tok(txt, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}, torch.tensor(ys, dtype=torch.long, device=device)

    @torch.no_grad()
    def match_acc(txts, ys):
        if len(txts) == 0:
            return float("nan")
        m.eval(); correct = 0
        for i in range(0, len(txts), 64):
            enc = tok(txts[i:i + 64], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = m(**enc).logits.argmax(-1).cpu().numpy()
            correct += int((pr == ys[i:i + 64]).sum())
        m.train(); return correct / len(txts)

    torch.manual_seed(0)
    m = load_model().to(device); m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batch, accum, lr = 16, 1, 1e-5
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(tr_texts) / (batch * accum)) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loader = DataLoader(DS(tr_texts, tr_y), batch_size=batch, shuffle=True, collate_fn=collate)
    m.train()
    for ep in range(epochs):
        opt.zero_grad(); seen = correct = nb = 0; lsum = 0.0
        for bi, (enc, yb) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(**enc).logits
                loss = torch.nn.functional.cross_entropy(logits, yb) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            with torch.no_grad():
                correct += int((logits.argmax(-1) == yb).sum().item()); seen += len(yb)
                lsum += loss.item() * accum; nb += 1
            if bi % 500 == 0:
                ha = match_acc(ho_texts, ho_y)
                log(f"  ep{ep} step{bi}/{len(loader)} loss={lsum/max(nb,1):.4f} train_acc={correct/max(seen,1):.4f} "
                    f"HELDOUT_acc={ha:.4f} ({int(time.time()-t0)}s)")
        ha = match_acc(ho_texts, ho_y)
        log(f"=== ep{ep} DONE train_acc={correct/max(seen,1):.4f} HELDOUT_acc={ha:.4f} "
            f"avg_loss={lsum/max(nb,1):.4f} ({int(time.time()-t0)}s) ===")

    os.makedirs(OUT_DIR, exist_ok=True)
    m.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
    log(f"CONTRASTIVE_DONE saved {OUT_DIR}  (set MBERT_DIR={OUT_DIR} RESET_HEAD=1 for the OOF)")


if __name__ == "__main__":
    main()
