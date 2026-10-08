"""Registered sweeps (protocol.md). Append-only JSON results under results/.
--smoke: plumbing check at pool 16 / 3 epochs / m=1.0, marked smoke.json."""
import json
import os
import subprocess
import sys
import time

import numpy as np

import embed_client
import heads
import hierarchy as H
import walks

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            text=True).strip()
    except Exception:
        return "unknown"


def prep(m):
    """Embed masters for one markedness config; returns (Xtr, ytr, Xev, yev)."""
    masters = H.build_masters(m)
    cache = embed_client.load_cache()
    texts = [t for t, _ in masters["train"]] + [t for t, _ in masters["eval"]]
    log(f"m={m}: embedding {len(set(texts) - set(cache))} new / "
        f"{len(set(texts))} texts")
    embed_client.embed(texts, cache)

    def mat(split):
        pairs = masters[split]
        X = np.stack([cache[t].astype(np.float32) for t, _ in pairs])
        y = np.array([i for _, i in pairs])
        return X, y
    return (*mat("train"), *mat("eval"))


def subsample(X, y, pool, seed):
    rng = np.random.default_rng(H.DATA_SEED * 7919 + pool * 104729 + seed)
    keep = np.concatenate([
        rng.choice(np.nonzero(y == lf)[0], size=pool, replace=False)
        for lf in np.unique(y)])
    keep.sort()
    return X[keep], y[keep]


def eval_all(models, Xev, yev, depths, oracle_arms=("C", "C'"), beam_from="B"):
    out = {}
    for d in depths:
        if "A" in models and d in models["A"]:
            out[f"A_d{d}"] = walks.flat_eval(models["A"][d], Xev, yev, d)
    for arm in ("B", "C", "C'"):
        if arm not in models:
            continue
        for d in depths:
            r = walks.hard_walk(models[arm], Xev, yev, d)
            if arm in oracle_arms:
                r["oracle_path_acc"] = walks.hard_walk(
                    models[arm], Xev, yev, d, oracle=True)["path_acc"]
            out.setdefault(arm, {})[f"d{d}"] = r
    if beam_from in models:
        for d in depths:
            out.setdefault("E", {})[f"d{d}"] = walks.beam2_walk(
                models[beam_from], Xev, yev, d)
    return out


def run_config(tag, m, pool, depths, arms, extra_seeds=(), epochs=40,
               target_depths_A=()):
    Xtr, ytr, Xev, yev = prep(m)
    Xp, yp = subsample(Xtr, ytr, pool, seed=0)
    log(f"{tag}: pool={pool} train={len(yp)} eval={len(yev)} arms={arms}")

    def log_train(msg):
        log(msg)

    models = {}
    for arm in arms:
        if arm == "A":
            models["A"] = {}
            for d in target_depths_A:
                t0 = time.time()
                models["A"][d] = heads.train("A", Xp, yp, seed=0,
                                             epochs=epochs,
                                             target_depth=d, log=log_train)
                log(f"{tag}: A_d{d} trained valCE={models['A'][d]['val_ce']:.4f} "
                    f"({time.time() - t0:.0f}s)")
        else:
            t0 = time.time()
            models[arm] = heads.train(arm, Xp, yp, seed=0, epochs=epochs,
                                      log=log_train)
            log(f"{tag}: {arm} trained valCE={models[arm]['val_ce']:.4f} "
                f"({time.time() - t0:.0f}s)")

    results = {"tag": tag, "m": m, "pool": pool, "depths": depths,
               "extra_seeds": list(extra_seeds), "commit": git_commit(),
               "model_fp": embed_client.fingerprint(),
               "n_train": int(len(yp)), "n_eval": int(len(yev)),
               "epochs": epochs, "arms": eval_all(models, Xev, yev, depths)}

    # extra head seeds (robustness registration): full walk record
    if extra_seeds:
        results["seed_runs"] = {}
        for s in extra_seeds:
            for arm in ("B", "C"):
                if arm not in arms:
                    continue
                mdl = heads.train(arm, Xp, yp, seed=s, epochs=epochs)
                results["seed_runs"][f"{arm}_seed{s}"] = {
                    f"d{d}": walks.hard_walk(mdl, Xev, yev, d)
                    for d in depths}
    return results


def save(tag, payload):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    log(f"wrote {path}")


def summary(tag, r):
    a = r["arms"]
    line = [tag]
    for d in r["depths"]:
        parts = []
        for k in ("A", "B", "C", "C'", "E"):
            if k == "A":
                if f"A_d{d}" in a:
                    parts.append(f"A={a[f'A_d{d}']['acc_leaf']:.3f}")
            elif k in a and f"d{d}" in a[k]:
                v = a[k][f"d{d}"]
                key = ("acc_leaf" if k == "A" else
                       "acc_final" if k == "E" else "path_acc")
                s = f"{k}={v[key]:.3f}"
                if "oracle_path_acc" in v:
                    s += f"(orc {v['oracle_path_acc']:.3f})"
                parts.append(s)
        line.append(f"d{d}: " + " ".join(parts))
    print(" | ".join(line), flush=True)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    if smoke:
        r = run_config("smoke", m=1.0, pool=16, depths=(2,), arms=("A", "B",
                                                                      "C"),
                       epochs=3, target_depths_A=(2,))
        save("smoke", r)
        summary("smoke", r)
    else:
        # SWEEP-1: m=1.0, depth 2/3/4, pool {16,64,256,1024}, all arms
        for pool in (16, 64, 256, 1024):
            tag = f"sweep1_pool{pool}"
            r = run_config(tag, m=1.0, pool=pool, depths=(2, 3, 4),
                           arms=("A", "B", "C", "C'"),
                           extra_seeds=(1, 2) if pool == 1024 else (),
                           target_depths_A=(2, 3, 4))
            save(tag, r)
            summary(tag, r)
        # SWEEP-2: depth 3, pool 256, m {0.5, 0.0}, arms B, C
        for m in (0.5, 0.0):
            tag = f"sweep2_m{str(m).replace('.', '')}"
            r = run_config(tag, m=m, pool=256, depths=(3,), arms=("B", "C"))
            save(tag, r)
            summary(tag, r)
    log(f"total {time.time() - t0:.0f}s")
