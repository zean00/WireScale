#!/usr/bin/env python
"""MS-24 runner: direct tool-score band selector (no clusters).

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-24) — committed before any scored run.
Shortlist = all tools whose cosine to the query is within DELTA of the
top scorer (band08), optionally capped at CAP by rank when the band
degenerates (band08cap40). No k-means for the band arms; the union arm
(shipped gate) reruns first as the sanity anchor. Imports frozen
machinery verbatim; writes results_band.jsonl / summary_band.json.

    python tools/mcp_probe/run_band.py --check   # constants only
    python tools/mcp_probe/run_band.py           # full pass (union first)
    MCP_BAND_CANARY=1 python tools/mcp_probe/run_band.py   # 6-case canary
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "..", "web")
sys.path.insert(0, WEB)
sys.path.insert(0, HERE)
os.chdir(WEB)
import server  # noqa: E402
from run_eval import embed, kmeans, load_tools, l2, plan, schema_ok  # noqa: E402
from run_beam import directive_for, shortlist  # noqa: E402

RESULTS = os.path.join(HERE, "results_band.jsonl")
SUMMARY = os.path.join(HERE, "summary_band.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5
DELTA = 0.08                # registered band width
CAP = 40                    # registered tail cap
SANITY_ACC = 0.6763
SANITY_TOL = 0.005
SANITY_NEG = 20
UNION_TOK = 2265            # union mean tokens (MS-23 scored run)
BUDGET_TOK_CAP = int(0.70 * UNION_TOK)   # MS-24b: 1585
ARMS = [("band08", False), ("band08cap40", True)]


def band_indices(sims, cap):
    top1 = float(sims.max())
    idx = np.where(sims >= top1 - DELTA)[0]
    if cap and len(idx) > CAP:
        keep = idx[np.argsort(-sims[idx], kind="stable")[:CAP]]
        return sorted(int(i) for i in keep), len(idx), True
    return sorted(int(i) for i in idx), len(idx), False


def main():
    ap = "--check" in sys.argv
    tools = load_tools()
    tools_by_name = {t["name"]: t for t in tools}
    n = len(tools)
    if ap:
        print(f"band selector: delta={DELTA}, cap={CAP}, {n} tools, "
              f"arms={['union'] + [a for a, _ in ARMS]}; "
              f"budget cap {BUDGET_TOK_CAP} tok (0.70 x union {UNION_TOK})")
        return 0

    cases = json.load(open(CASES))
    if os.environ.get("MCP_BAND_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{n} tools, {len(cases)} cases, "
          f"arms={['union'] + [a for a, _ in ARMS]}")
    C = json.load(open(BUDGET))["C"]

    pools = json.load(open(POOLS))
    t0 = time.time()
    svecs = []
    for t in tools:
        us = pools.get(f"{t['server']}__{t['name']}", [])[:8]
        svecs.append(l2(np.stack([l2(embed(u)) for u in us]).mean(0)))
    vecs = np.stack(svecs)
    cents, assign = kmeans(vecs, k=K)
    space = {"cents": cents, "assign": assign, "vecs": vecs}
    print(f"space built in {time.time()-t0:.0f}s", flush=True)

    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                done.add((r["arm"], r["case_id"]))
            except (ValueError, KeyError):
                pass
    out = open(RESULTS, "a")
    t_start = time.time()

    for arm, cap in [("union", False)] + list(ARMS):
        n_done = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            v = l2(embed(c["utterance"]))
            if arm == "union":
                idx, clusters = shortlist(space, v, 2, None)
                route = {"clusters": clusters}
            else:
                sims = vecs @ v
                idx, band_size, capped = band_indices(sims, cap)
                route = {"band_size": int(band_size), "capped": capped}
            short = [tools[i] for i in idx]
            route["shortlist"] = len(short)
            stripped = [tools[i]["name"] for i in idx]
            gold_rank = stripped.index(c["want"]) + 1 \
                if c["want"] in stripped else None
            d = directive_for(short, C)
            r = plan(c["utterance"], d)
            rec = {"arm": arm, "case_id": c["id"], "kind": c["kind"],
                   "want": c["want"], **r,
                   "route": {**route, "gold_rank": gold_rank}}
            if "plan" in r and r["plan"] is not None:
                got = r["plan"]["tool"]
                rec["got_raw"] = got
                if isinstance(got, str):
                    got = got.rsplit("__", 1)[-1]
                rec["got"] = got
                rec["ok"] = (got == c["want"]) if c["want"] is not None \
                    else (got is None)
                if rec["ok"] and c["want"] is not None:
                    rec["schema_ok"] = schema_ok(got, r["plan"].get("args"),
                                                 tools_by_name)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f"  {arm}: {n_done} new cases, {el/60:.1f} min elapsed",
                      flush=True)
        print(f"arm {arm} done (+{n_done} cases)", flush=True)
        if arm == "union" and not os.environ.get("MCP_BAND_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO band "
                      "arms run (protocol MS-24: no verdict on drift). "
                      "Fix the baseline first.")
                return 1
    summarize()
    return 0


def summarize_arm(arm):
    a = {"n": 0, "ok": 0, "pt": [], "ms": [], "pos": [0, 0], "neg": [0, 0]}
    for line in open(RESULTS):
        r = json.loads(line)
        if r["arm"] != arm or "error" in r:
            continue
        a["n"] += 1
        a["ok"] += int(r.get("ok", False))
        a["pt"].append(r.get("prompt_tokens", 0))
        a["ms"].append(r.get("ms", 0))
        if r["kind"] in ("pos", "trap"):
            a["pos"][0] += int(r.get("ok", False))
            a["pos"][1] += 1
        else:
            a["neg"][0] += int(r.get("ok", False))
            a["neg"][1] += 1
    return {"n": a["n"], "acc": round(a["ok"] / max(a["n"], 1), 4),
            "pos": a["pos"][0], "pos_n": a["pos"][1],
            "neg": a["neg"][0], "neg_n": a["neg"][1],
            "pt": round(sum(a["pt"]) / max(len(a["pt"]), 1), 1),
            "ms": round(sum(a["ms"]) / max(len(a["ms"]), 1), 1)}


def summarize():
    arms = {}
    for line in open(RESULTS):
        r = json.loads(line)
        a = arms.setdefault(r["arm"], {"n": 0, "ok": 0, "pt": [], "ms": [],
                                       "schema": [], "pos": [0, 0],
                                       "neg": [0, 0], "hit": [0, 0],
                                       "phit": [0, 0]})
        if "error" in r:
            continue
        a["n"] += 1
        a["ok"] += int(r.get("ok", False))
        a["pt"].append(r.get("prompt_tokens", 0))
        a["ms"].append(r.get("ms", 0))
        if r.get("ok") and "schema_ok" in r:
            a["schema"].append(int(r["schema_ok"]))
        hit = r.get("route", {}).get("gold_rank") is not None
        if r["kind"] in ("pos", "trap"):
            a["pos"][0] += int(r.get("ok", False))
            a["pos"][1] += 1
            a["hit"][0] += int(hit)
            a["hit"][1] += 1
            if hit:
                a["phit"][0] += int(r.get("ok", False))
                a["phit"][1] += 1
        else:
            a["neg"][0] += int(r.get("ok", False))
            a["neg"][1] += 1
    summary = {}
    for arm, a in arms.items():
        summary[arm] = {
            "n": a["n"], "acc": round(a["ok"] / max(a["n"], 1), 4),
            "pos_acc": round(a["pos"][0] / max(a["pos"][1], 1), 4),
            "neg": f"{a['neg'][0]}/{a['neg'][1]}",
            "schema_ok": round(sum(a["schema"]) / len(a["schema"]), 4)
            if a["schema"] else None,
            "hit_rate": round(a["hit"][0] / max(a["hit"][1], 1), 4),
            "pos_acc_given_hit": round(a["phit"][0] / max(a["phit"][1], 1), 4)
            if a["phit"][1] else None,
            "mean_prompt_tokens": round(sum(a["pt"]) / max(len(a["pt"]), 1), 1),
            "mean_ms": round(sum(a["ms"]) / max(len(a["ms"]), 1), 1)}
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if "union" in summary and not os.environ.get("MCP_BAND_CANARY"):
        u = summary["union"]
        for arm, _ in ARMS:
            if arm not in summary:
                continue
            b = summary[arm]
            neg_ok = int(b["neg"].split("/")[0]) >= SANITY_NEG
            ok_a = b["acc"] >= u["acc"] + 0.02 and neg_ok
            print(f"MS-24a {arm}: acc {b['acc']} vs union+0.02 "
                  f"{round(u['acc'] + 0.02, 4)}, neg {b['neg']} -> "
                  f"{'ADOPT AS GATE' if ok_a else 'KEEP UNION'}")
            ok_b = (b["acc"] >= u["acc"] - 0.01
                    and b["mean_prompt_tokens"] <= BUDGET_TOK_CAP)
            print(f"MS-24b {arm}: acc {b['acc']} vs union-0.01 "
                  f"{round(u['acc'] - 0.01, 4)}, tokens "
                  f"{b['mean_prompt_tokens']} <= {BUDGET_TOK_CAP} -> "
                  f"{'BUDGET PARETO (gate accuracy at ~half cost)' if ok_b else 'no budget win'}")


if __name__ == "__main__":
    sys.exit(main())
