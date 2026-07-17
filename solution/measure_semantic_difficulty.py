"""DeepSeek idea #6: semantic objective difficulty.
Ridge regression maps a learning-objective's MiniLM embedding -> its average
correctness, with leave-one-objective-out OOF so it transfers to UNSEEN
objectives (a text->difficulty FUNCTION, not leaky per-objective memorization).
A/B: numeric HGB objective-grouped, with vs without this feature.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from features import numeric_feature_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def L(y, p): return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def main():
    from sentence_transformers import SentenceTransformer
    X = pd.read_parquet(os.path.join(CACHE, "train_X.parquet"))
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[X.index, "is_correct"].to_numpy(float)
    obj = f.loc[X.index, "learning_objective_id"].astype(str).to_numpy()
    obj_text = f.loc[X.index, "learning_objective"].astype(str)

    # unique objectives -> text + mean correctness
    uo = pd.DataFrame({"obj": obj, "y": y, "text": obj_text.values}).groupby("obj").agg(
        text=("text", "first"), ymean=("y", "mean"), n=("y", "size")).reset_index()
    print(f"{len(uo)} unique objectives", flush=True)
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(uo.text.tolist(), normalize_embeddings=True)

    # leave-one-objective-out (KFold over objectives) OOF ridge difficulty
    oof = np.zeros(len(uo))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(emb):
        r = Ridge(alpha=10.0).fit(emb[tr], uo.ymean.values[tr])
        oof[va] = r.predict(emb[va])
    diff_map = dict(zip(uo.obj, oof))
    sem_diff = np.array([diff_map.get(o, uo.ymean.mean()) for o in obj])
    print(f"semantic-difficulty corr with objective mean-correctness: "
          f"{np.corrcoef(oof, uo.ymean)[0,1]:.3f}", flush=True)

    num = X[numeric_feature_columns(X)].to_numpy(float)
    med = np.nanmedian(num, 0); med = np.where(np.isnan(med), 0, med)
    num = np.where(np.isnan(num), med, num)
    groups = obj

    def cv(M):
        oofp = np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(M, y, groups):
            clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                    l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, random_state=0)
            clf.fit(M[tr], y[tr]); oofp[va] = clf.predict_proba(M[va])[:, 1]
        return L(y, oofp), roc_auc_score(y, oofp)

    b_ll, b_auc = cv(num)
    w_ll, w_auc = cv(np.hstack([num, sem_diff.reshape(-1, 1)]))
    print(f"  without semantic-difficulty: ll={b_ll:.5f} auc={b_auc:.4f}")
    print(f"  with semantic-difficulty   : ll={w_ll:.5f} auc={w_auc:.4f}")
    print(f"  effect: {w_auc-b_auc:+.4f} AUC, {b_ll-w_ll:+.5f} logloss")
    print("  => HELPS" if w_auc > b_auc + 0.001 else "  => no clear gain")


if __name__ == "__main__":
    main()
