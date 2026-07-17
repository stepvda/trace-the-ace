"""Which signals TRANSFER across a domain shift? Leak-free leave-one-domain-out.

Hypothesis: transcript TF-IDF (provider-specific vocabulary) fails to transfer /
reverses under the provider shift, dragging the model below the constant; numeric
behavioral features + objective-text TF-IDF (provider-invariant) transfer fine.

For each held-out style-domain we REFIT every text vectorizer on the training
domains only (no leakage), so the measured transfer is honest. We compare each
signal's out-of-domain logloss gain over that domain's constant.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
import shift_proxy as S

def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def lr_text(train_txt, y_tr, test_txt, maxf=20000):
    v = TfidfVectorizer(max_features=maxf, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    Xtr = v.fit_transform(train_txt); Xte = v.transform(test_txt)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(Xtr, y_tr)
    return lr.predict_proba(Xte)[:, 1]


def hgb_num(num_tr, y_tr, num_te):
    c = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, random_state=0)
    c.fit(num_tr, y_tr); return c.predict_proba(num_te)[:, 1]


def main():
    X, num, svd, y, obj = S.load()
    dom = S.make_domains(num, svd, k=6)
    text_all = X["text_all"].fillna("").to_numpy()
    text_lo = X["text_lo"].fillna("").to_numpy()
    signals = ["numeric", "transcript_tfidf", "objective_tfidf", "num+obj", "num+transcript(FULL)"]
    res = {s: [] for s in signals}
    aucs = {s: [] for s in signals}
    for d in sorted(set(dom)):
        tr, te = dom != d, dom == d
        if te.sum() < 100:
            continue
        prior = y[tr].mean(); c = np.full(te.sum(), prior)
        llc = L(y[te], c)
        pn = hgb_num(num[tr], y[tr], num[te])
        ptr = lr_text(text_all[tr], y[tr], text_all[te])
        plo = lr_text(text_lo[tr], y[tr], text_lo[te], maxf=3000)
        combos = {
            "numeric": pn,
            "transcript_tfidf": ptr,
            "objective_tfidf": plo,
            "num+obj": 0.5 * pn + 0.5 * plo,
            "num+transcript(FULL)": 0.5 * pn + 0.5 * ptr,
        }
        for s, p in combos.items():
            res[s].append(llc - L(y[te], p))
            aucs[s].append(roc_auc_score(y[te], p))
    print(f"\n=== out-of-domain TRANSFER (leak-free, {len(res['numeric'])} domains) ===")
    print(f"{'signal':<22} {'gain_vs_const':>14} {'mean_auc':>9}")
    for s in signals:
        print(f"{s:<22} {np.mean(res[s]):>+14.4f} {np.mean(aucs[s]):>9.4f}")
    print("\n(positive gain => beats the constant out-of-domain => likely to transfer)")


if __name__ == "__main__":
    main()
