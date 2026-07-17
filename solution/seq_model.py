"""KT-style sequence model: GRU over per-turn dialogue features + transcript/objective
SVD globals -> P(next answer correct). Objective-grouped 80/20; saves OOF for ensembling.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def main(svd_dim=64, hidden=48, epochs=10, batch=128, lr=8e-4):
    import torch, torch.nn as nn
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import log_loss, roc_auc_score
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    seq_X = np.load(os.path.join(CACHE, "seq_X.npy"))
    sids = pd.read_csv(os.path.join(CACHE, "seq_sids.csv")).iloc[:, 0].astype(str).tolist()
    sid2idx = {s: i for i, s in enumerate(sids)}
    ids = pd.read_csv(os.path.join(CACHE, "row_ids.csv")).iloc[:, 0].astype(str).tolist()
    svd = np.load(os.path.join(CACHE, "svd256.npy"))[:, :svd_dim].astype(np.float32)
    svd = (svd - svd.mean(0)) / (svd.std(0) + 1e-6)
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv")).set_index("response_id")
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    y = lab.loc[ids, "is_correct"].to_numpy(np.float32)
    groups = f.loc[ids, "learning_objective_id"].astype(str).to_numpy()
    resp_sidx = np.array([sid2idx.get(str(f.loc[r, "session_id"]), 0) for r in ids])

    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(ids, y, groups))
    print(f"device={dev} train={len(tr)} val={len(va)} seq_X={seq_X.shape}", flush=True)
    seqT = torch.tensor(seq_X, device=dev)  # (n_sess, T, F)
    svdT = torch.tensor(svd, device=dev)
    yT = torch.tensor(y, device=dev)
    sidxT = torch.tensor(resp_sidx, device=dev)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(seq_X.shape[2], hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden + svd_dim, 48), nn.ReLU(),
                                      nn.Dropout(0.4), nn.Linear(48, 1))
        def forward(self, sidx, gvec):
            out, _ = self.gru(seqT[sidx])
            return self.head(torch.cat([out[:, -1, :], gvec], -1)).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    tr = np.array(tr); rng = np.random.RandomState(0)
    vy = y[va]
    best_ll, best_vp, best_ep = 9.9, None, -1
    t0 = time.time()
    for ep in range(epochs):
        net.train(); rng.shuffle(tr)
        for i in range(0, len(tr), batch):
            b = tr[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(sidxT[b], svdT[b]), yT[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vp = np.concatenate([torch.sigmoid(net(sidxT[va[i:i + 512]], svdT[va[i:i + 512]])).cpu().numpy()
                                 for i in range(0, len(va), 512)])
        ll = log_loss(vy, np.clip(vp, 1e-6, 1 - 1e-6)); au = roc_auc_score(vy, vp)
        print(f"  ep{ep} val logloss={ll:.5f} auc={au:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ll < best_ll:
            best_ll, best_vp, best_ep = ll, vp, ep
    pd.DataFrame({"response_id": np.array(ids)[va], "p_ft": best_vp, "y": vy}).to_csv(
        os.path.join(CACHE, "ft_val_seq.csv"), index=False)
    print(f"BEST ep{best_ep}: AUC={roc_auc_score(vy, best_vp):.4f} logloss={best_ll:.5f}", flush=True)


if __name__ == "__main__":
    main()
