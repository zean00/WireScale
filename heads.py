"""Trunk + per-level classifiers, arms A/B/C/C' (protocol.md).

Trunk: Linear(D->2048) -> GELU(tanh) -> LayerNorm -> Linear(->1024).
C concat: input = [emb | onehot(parent)] (root level = zeros).
C' FiLM: emb' = emb*(1+gamma)+beta, gamma/beta = zero-init Linear(onehot).
Training uses per-level linear CE heads (discarded afterwards); inference
uses L2-normalized class-mean centroids from train outputs, cosine argmax.
"""
import os
import numpy as np
import torch
import torch.nn as nn

import hierarchy as H

HIDDEN, OUT = 2048, 1024
N_NODES = 45

_NODES = H.node_table(6)
_ORDER = list(_NODES)
_IDX = {n: i for i, n in enumerate(_ORDER)}
_BY_DEPTH = {d: [n for n in _ORDER if _NODES[n]["depth"] == d]
             for d in (1, 2, 3, 4, 5, 6)}
_PATHS = [tuple(p) for p in H.leaf_paths(4)]
_LOCAL = {d: {n: j for j, n in enumerate(_BY_DEPTH[d])} for d in _BY_DEPTH}
N_NODES = len(_ORDER) - 1


def set_depth(tree_depth):
    """Switch label registries to the run's tree depth (4 or 6)."""
    global _PATHS
    _PATHS = [tuple(p) for p in H.leaf_paths(tree_depth)]
    H.set_depth(tree_depth)


def level_label(leaf_idx, depth):
    """Local class index within the depth-`depth` head."""
    return _LOCAL[depth][_PATHS[leaf_idx][:depth]]


def parent_label(leaf_idx, depth):
    """GLOBAL node index conditioning input for a readout AT `depth` (1..4):
    the gold node at depth-1; root level (depth 1) -> N_NODES sentinel."""
    return _IDX[H.class_at(depth - 1, leaf_idx)] if depth > 1 else N_NODES


def depth_node_order(depth):
    return _BY_DEPTH[depth]


class Net(nn.Module):
    def __init__(self, arm, levels=(1, 2, 3, 4)):
        super().__init__()
        self.arm = arm
        d = 2560 + (N_NODES if arm == "C" else 0)
        self.fc1 = nn.Linear(d, HIDDEN)
        self.act = nn.GELU(approximate="tanh")
        self.ln = nn.LayerNorm(HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, OUT)
        self.film = (nn.Linear(N_NODES, 2 * 2560) if arm == "C'" else None)
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.heads = nn.ModuleList(nn.Linear(OUT, len(_BY_DEPTH[d]))
                                   for d in levels)

    def embed(self, x, onehot):
        if self.arm == "C":
            x = torch.cat([x, onehot], dim=1)
        elif self.arm == "C'":
            gb = self.film(onehot)
            gamma, beta = gb[:, :2560], gb[:, 2560:]
            x = x * (1.0 + gamma) + beta
        return self.fc2(self.ln(self.act(self.fc1(x))))


def _t(a, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)


def onehot_matrix(node_idxs, n):
    """Root sentinel (N_NODES) -> zero row."""
    o = np.zeros((n, N_NODES), dtype=np.float32)
    ok = node_idxs < N_NODES
    o[np.nonzero(ok)[0], node_idxs[ok]] = 1.0
    return o


def train(arm, X, leaf_ids, seed, epochs=40, patience=6, lr=1e-3, batch=512,
          target_depth=None, levels=None, log=None):
    """X: (n,2560) fp32 train embeddings; leaf_ids: (n,). target_depth=None
    trains the full multi-task tree (B/C/C'); an int trains a flat head at
    that depth only (A). levels: which tree levels to train (default 1-4).
    Returns model dict."""
    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(1000 + seed)
    if target_depth:
        levels = (target_depth,)
    elif levels is None:
        levels = (1, 2, 3, 4)

    # stratified 10% val carve by leaf class
    val_mask = np.zeros(len(X), dtype=bool)
    for lf in np.unique(leaf_ids):
        idx = np.nonzero(leaf_ids == lf)[0]
        val_mask[rng.choice(idx, size=max(1, len(idx) // 10), replace=False)] = True
    tr, va = ~val_mask, val_mask

    labs = {d: np.array([level_label(l, d) for l in leaf_ids]) for d in levels}
    par = {d: np.array([parent_label(l, d) for l in leaf_ids]) for d in levels}

    net = Net(arm, levels)
    head_pos = {d: i for i, d in enumerate(levels)}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    Xt = _t(X)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(np.nonzero(tr)[0])
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = 0.0
            for d in levels:
                oh = torch.from_numpy(onehot_matrix(par[d][b], len(b)))
                logits = net.embed(Xt[b], oh)
                loss = loss + ce(net.heads[head_pos[d]](logits),
                                 _t(labs[d][b], torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b) / max(len(levels), 1)
        net.eval()
        with torch.no_grad():
            vce = 0.0
            for d in levels:
                ohv = torch.from_numpy(onehot_matrix(par[d][va], int(va.sum())))
                vlogits = net.embed(Xt[va], ohv)
                vce += float(ce(net.heads[head_pos[d]](vlogits),
                                _t(labs[d][va], torch.long)))
            vce /= max(len(levels), 1)
        if log:
            log(f"    ep{ep + 1} trainCE/tok {tot / tr.sum():.4f} valCE {vce:.4f}")
        if vce < best - 1e-4:
            best, bad = vce, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    # centroids from TRAIN outputs (val/eval excluded); gold parents
    with torch.no_grad():
        otr = np.zeros((int(tr.sum()), OUT), dtype=np.float32)
        for d in levels:
            oh = torch.from_numpy(onehot_matrix(par[d][tr], int(tr.sum())))
            otr += net.embed(Xt[tr], oh).numpy().astype(np.float32)
        otr /= max(len(levels), 1)
    cents = {}
    tr_leaf = leaf_ids[tr]
    for d in levels:
        names = depth_node_order(d)
        tl = np.array([level_label(l, d) for l in tr_leaf])
        M = np.zeros((len(names), OUT), dtype=np.float32)
        for j in range(len(names)):
            m = tl == j
            if m.any():
                mu = otr[m].mean(axis=0)
                M[j] = mu / max(float(np.linalg.norm(mu)), 1e-8)
        cents[d] = (names, M)
    return {"arm": arm, "seed": seed, "net": net, "centroids": cents,
            "val_ce": best, "levels": levels}


def trunk_out(model, X, node_idxs):
    """Trunk outputs; len(X) must equal len(node_idxs) (one row per
    (sample, conditioning) pair — callers repeat X for beams)."""
    net = model["net"]
    node_idxs = np.asarray(node_idxs)
    assert len(X) == len(node_idxs), (len(X), len(node_idxs))
    with torch.no_grad():
        oh = torch.from_numpy(onehot_matrix(node_idxs, len(X)))
        return net.embed(_t(X), oh).numpy().astype(np.float32)
