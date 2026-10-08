"""Walks + metrics: hard tree (B/C/C'), beam-2 (E), flat (A). Oracle-parent
variant of the hard walk gives the teacher-forcing bound for C/C'. Scores
are cosine vs per-node centroids, restricted to candidate children."""
import numpy as np

import heads
import hierarchy as H


def score_at(model, X, depth, parent_nodes):
    """(n_rows, n_candidates) cosine vs depth-`depth` centroids, one row per
    conditioning index (N_NODES sentinel = root, i.e. unconditioned)."""
    names, M = model["centroids"][depth]
    Y = heads.trunk_out(model, X, parent_nodes)
    return Y @ M.T, names


def _child_mask(names, parent):
    return np.array([n[:len(parent)] == parent for n in names], dtype=bool)


def _gold_nodes(leaf_ids, depth):
    return np.array([[heads.level_label(l, d) for d in range(1, depth + 1)]
                     for l in leaf_ids])


def hard_walk(model, X, leaf_ids, depth, oracle=False):
    """Greedy top-1 per level with restriction to the chosen prefix."""
    n = len(X)
    gold = _gold_nodes(leaf_ids, depth)  # local per-level class indices
    gold_global = np.array([[heads._IDX[H.class_at(d, l)] for d in
                             range(1, depth + 1)] for l in leaf_ids])
    correct = {d: np.zeros(n, dtype=bool) for d in range(1, depth + 1)}
    parents = np.full(n, heads.N_NODES)
    chosen, prev_names = None, None
    for d in range(1, depth + 1):
        S, names = score_at(model, X, d, parents)
        pick = np.zeros(n, dtype=np.int64)
        for i in range(n):
            mask = (np.ones(len(names), dtype=bool) if d == 1
                    else _child_mask(names, prev_names[chosen[i]]))
            pick[i] = int(np.argmax(np.where(mask, S[i], -np.inf)))
        chosen, prev_names = pick, names
        correct[d] = pick == gold[:, d - 1]
        if oracle:
            parents = gold_global[:, d - 1].copy()
        else:
            parents = np.array([heads._IDX[names[j]] for j in pick])
    metrics = {}
    for d in range(1, depth + 1):
        base = (np.ones(n, dtype=bool) if d == 1
                else np.logical_and.reduce([correct[b] for b in range(1, d)]))
        metrics[f"acc_L{d}"] = float((correct[d] & base).sum()
                                     / max(int(base.sum()), 1))
    metrics["path_acc"] = float(np.logical_and.reduce(
        [correct[d] for d in range(1, depth + 1)]).mean())
    metrics["chance_path"] = (1 / 3) * (0.5 ** (depth - 1))
    return metrics


def beam2_walk(model, X, leaf_ids, depth):
    """E: keep the top-2 (node, cumulative cosine) pairs per level; the final
    level's argmax over the frontier is the answer."""
    n = len(X)
    gold = _gold_nodes(leaf_ids, depth)
    S, names = score_at(model, X, 1, np.full(n, heads.N_NODES))
    frontier = np.argsort(-S, axis=1)[:, :2]
    fscore = np.take_along_axis(S, frontier, axis=1)
    for d in range(2, depth + 1):
        parent_ids = np.array([[heads._IDX[names[j]] for j in row]
                               for row in frontier])
        Sc, cnames = score_at(model, np.repeat(X, 2, axis=0), d,
                              parent_ids.reshape(-1))
        Sc = Sc.reshape(n, 2, -1)
        cnd = np.zeros((n, 2, len(cnames)), dtype=bool)
        for k in range(2):
            for i in range(n):
                cnd[i, k] = _child_mask(cnames, names[frontier[i, k]])
        cS = np.where(cnd, Sc, -np.inf).reshape(n, -1)
        cF = np.repeat(fscore, len(cnames), axis=1) + cS
        top = np.argsort(-cF, axis=1)[:, :2]
        frontier = np.tile(np.arange(len(cnames)), 2)[top]
        fscore = np.take_along_axis(cF, top, axis=1)
        names = cnames
    truth = gold[:, depth - 1]
    return {"acc_final": float((frontier[:, 0] == truth).mean()),
            "chance_final": (1 / 3) * (0.5 ** (depth - 1))}


def flat_eval(model, X, leaf_ids, depth):
    """A: single argmax over depth-d centroids; leaf accuracy is the metric."""
    names, M = model["centroids"][depth]
    Y = heads.trunk_out(model, X, np.full(len(X), heads.N_NODES))
    pred = np.argmax(Y @ M.T, axis=1)
    truth = np.array([heads.level_label(l, depth) for l in leaf_ids])
    return {"acc_leaf": float((pred == truth).mean()),
            "n_classes": len(names), "chance_leaf": 1 / len(names)}
