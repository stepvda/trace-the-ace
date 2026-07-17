"""Precompute & cache the expensive transforms ONCE so overnight experiments are
cheap: raw TF-IDF (sparse), SVD(256) dense, and the scaled numeric matrix — all
aligned to the row order of cache/train_X.parquet.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from features import numeric_feature_columns

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def main():
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    num_cols = numeric_feature_columns(X)
    np.save(os.path.join(CACHE, "num.npy"), X[num_cols].to_numpy(np.float32))
    pd.Series(num_cols).to_csv(os.path.join(CACHE, "num_cols.csv"), index=False)
    pd.Series(X.index).to_csv(os.path.join(CACHE, "row_ids.csv"), index=False)

    t0 = time.time()
    v_all = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=5,
                            sublinear_tf=True, strip_accents="unicode").fit(X["text_all"].fillna(""))
    Xa = v_all.transform(X["text_all"].fillna(""))
    v_st = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5,
                           sublinear_tf=True, strip_accents="unicode").fit(X["text_student"].fillna(""))
    Xs = v_st.transform(X["text_student"].fillna(""))
    # objective-description tfidf: captures generalizable TOPIC difficulty
    v_lo = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, strip_accents="unicode").fit(X["text_lo"].fillna(""))
    Xlo = v_lo.transform(X["text_lo"].fillna(""))
    T = sparse.hstack([Xa, Xs, Xlo]).tocsr()
    sparse.save_npz(os.path.join(CACHE, "tfidf.npz"), T)
    print(f"tfidf {T.shape} in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    svd = TruncatedSVD(n_components=256, random_state=0, n_iter=6).fit(T)
    Xsvd = svd.transform(T).astype(np.float32)
    np.save(os.path.join(CACHE, "svd256.npy"), Xsvd)
    print(f"svd256 in {time.time()-t0:.1f}s explained_var={svd.explained_variance_ratio_.sum():.3f}", flush=True)
    print("cached: num.npy, tfidf.npz, svd256.npy, row_ids.csv", flush=True)


if __name__ == "__main__":
    main()
