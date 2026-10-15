"""S-amendment: paired significance for the answer-level season
(protocol.md, registered pre-computation).

Mechanical uncertainty for the six paired headline gates: exact
McNemar + paired bootstrap (10,000 resamples, seed 0) on committed
per-question records. No model calls; annotates, never
re-adjudicates.

    python stats.py    # recompute from results/ (deterministic)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NBOOT = 10_000
SEED = 0


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for b=(A right, B wrong), c=vice versa."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(vec_a, vec_b, keys):
    """Return per-question 0/1 vectors aligned on keys."""
    return (np.array([vec_a[k] for k in keys]),
            np.array([vec_b[k] for k in keys]))


def analyze(name, a, b):
    """a, b: 0/1 arrays; report acc diff a-b, McNemar, bootstrap CI."""
    n = len(a)
    acc_a, acc_b = float(a.mean()), float(b.mean())
    b_, c_ = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = mcnemar_exact(b_, c_)
    rng = np.random.default_rng(SEED)
    diffs = np.array([a[rng.integers(0, n, n)].mean()
                      - b[rng.integers(0, n, n)].mean()
                      for _ in range(NBOOT)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"comparison": name, "n": n, "acc_a": round(acc_a, 4),
            "acc_b": round(acc_b, 4), "diff": round(acc_a - acc_b, 4),
            "discordant": {"a_only": b_, "b_only": c_},
            "mcnemar_p": round(p, 4),
            "boot_ci95": [round(float(lo), 4), round(float(hi), 4)]}


def main():
    asm = {(r["doc"], r["q"]): r for r in
           json.load(open(os.path.join(RES, "assembly.json")))["records"]}
    topk = {(r["doc"], r["q"]): r for r in
            json.load(open(os.path.join(RES, "topk.json")))["records"]}
    pool = {(r["doc"], r["q"]): r for r in
            json.load(open(os.path.join(RES, "poolstop.json")))["records"]}
    keys = sorted(asm.keys())
    assert set(topk) == set(asm) == set(pool), "record keys diverge"

    def ok(rec, arm):
        return int(rec[arm]["ans_ok"])

    runs = [
        ("S-A1  ASM vs TOP1", [ok(asm[k], "asm") for k in keys],
         [ok(asm[k], "top1") for k in keys]),
        ("S-A2  ASM vs FLATPOL", [ok(asm[k], "asm") for k in keys],
         [ok(asm[k], "flatpol") for k in keys]),
        ("S-B1  ASM vs TOPK-3", [ok(asm[k], "asm") for k in keys],
         [ok(topk[k], "topk3") for k in keys]),
        ("S-B2  ASM vs TOPK-2", [ok(asm[k], "asm") for k in keys],
         [ok(topk[k], "topk2") for k in keys]),
        ("S-C1  ASM-8 vs ASM", [ok(pool[k], "asm8") for k in keys],
         [ok(asm[k], "asm") for k in keys]),
        ("S-C3  ASM-8 vs TOPK-2", [ok(pool[k], "asm8") for k in keys],
         [ok(topk[k], "topk2") for k in keys]),
    ]
    out = {"tag": "stats_amendment", "nboot": NBOOT, "seed": SEED,
           "results": [analyze(n, np.array(a, int), np.array(b, int))
                       for n, a, b in runs]}
    # S-C1 registered non-inferiority read: CI lower bound vs -0.02
    c1 = out["results"][4]
    lo = c1["boot_ci95"][0]
    c1["solid_ni"] = ("SOLID-NI" if lo >= -0.02 else
                      "SUGGESTIVE-NI" if lo >= -0.04 else "NOT ESTABLISHED")
    path = os.path.join(RES, "stats_amendment.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
