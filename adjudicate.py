"""Mechanical adjudication of P1-P4 against the registered decision rules.
Reads results/*.json; prints the verdict block. No judgment calls here —
every threshold is quoted from protocol.md."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "results", f"{tag}.json")) as f:
        return json.load(f)


def pathacc(r, arm, d, key="path_acc"):
    a = r["arms"][arm]
    return a[f"d{d}"][key] if d else None


def main():
    s1 = {p: load(f"sweep1_pool{p}") for p in (16, 64, 256, 1024)}
    m0 = load("sweep2_m00")
    m05 = load("sweep2_m05")
    out = []

    # P1: B path-acc > A flat by >= 3 pts at d=3, m=1.0, pool>=64
    p1 = []
    for p in (64, 256, 1024):
        r = s1[p]
        b = pathacc(r, "B", 3)
        a = r["arms"][f"A_d3"]["acc_leaf"]
        p1.append((p, a, b, b - a))
    ok1 = all(g >= 0.03 for _, _, _, g in p1)
    out.append(("P1", ok1, p1))

    # P2: (C-B) at 1024 exceeds (C-B) at 16 by >= 3 pts (d=4, predicted)
    g16 = pathacc(s1[16], "C", 4) - pathacc(s1[16], "B", 4)
    g1024 = pathacc(s1[1024], "C", 4) - pathacc(s1[1024], "B", 4)
    ok2 = (g16 <= 0.03) and (g1024 - g16 >= 0.03)
    out.append(("P2", ok2, {"gap_pool16": g16, "gap_pool1024": g1024}))

    # P3: at m=0.0, deepest-level conditional acc within +-5 pts of 50% (B,C)
    p3 = {}
    for name, r in (("B", m0), ("C", m0)):
        p3[name] = r["arms"][name]["d3"]["acc_L3"]
    ok3 = all(abs(v - 0.5) <= 0.05 for v in p3.values())
    out.append(("P3", ok3, p3))

    # P4 (primary): at pool 1024, (C-B) gap at d4 minus gap at d2 >= +5 pts
    b2, b4 = pathacc(s1[1024], "B", 2), pathacc(s1[1024], "B", 4)
    c2, c4 = pathacc(s1[1024], "C", 2), pathacc(s1[1024], "C", 4)
    o2 = pathacc(s1[1024], "C", 2, "oracle_path_acc")
    o4 = pathacc(s1[1024], "C", 4, "oracle_path_acc")
    delta_pred = (c4 - b4) - (c2 - b2)
    delta_orc = (o4 - b4) - (o2 - b2)
    seeds = s1[1024].get("seed_runs", {})
    spread = {k: {"d2": v["d2"]["path_acc"], "d4": v["d4"]["path_acc"]}
              for k, v in seeds.items()}
    if delta_pred >= 0.05:
        verdict = "PASS"
    elif delta_orc >= 0.05:
        verdict = "PARTIAL (oracle-parent only)"
    elif delta_pred <= 0.0:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL (gap grows <5 pts)"
    out.append(("P4", verdict, {
        "B_d2": b2, "B_d4": b4, "C_d2": c2, "C_d4": c4,
        "C_oracle_d2": o2, "C_oracle_d4": o4,
        "delta_pred": delta_pred, "delta_oracle": delta_orc,
        "seed_spread": spread}))

    # context: E vs B at pool 1024
    e4 = s1[1024]["arms"]["E"]["d4"]["acc_final"]
    out.append(("E-vs-B@1024", None, {"E_d4": e4, "B_d4": b4}))

    for name, ok, detail in out:
        print(f"== {name}: {ok if ok is not None else ''}")
        print(json.dumps(detail, indent=1, default=float))


if __name__ == "__main__":
    main()
