#!/usr/bin/env python
"""MS-23 runner: window-union combination ("safety net").

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-23) — committed before any scored run.
Shortlist = k-means top-2 union (the shipped gate) UNION sliding-window
picks over the full registry (MS-22 tiling: window 10, step 7, append
rule; score = max member cosine). Arms: uwin2 (top-2 windows) and
uwinthr (threshold band top-0.03). Imports frozen machinery verbatim;
writes results_uwin.jsonl / summary_uwin.json.

    python tools/mcp_probe/run_uwin.py --check   # shapes only
    python tools/mcp_probe/run_uwin.py           # full pass (union first)
    MCP_UWIN_CANARY=1 python tools/mcp_probe/run_uwin.py   # 6-case canary
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

RESULTS = os.path.join(HERE, "results_uwin.jsonl")
SUMMARY = os.path.join(HERE, "summary_uwin.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5
WIN = 10
STEP = 7
DELTA = 0.03
SANITY_ACC = 0.6763
SANITY_TOL = 0.005
SANITY_NEG = 20
FLAT_ACC = 0.6835          # frozen flat-121 baseline (summary.json)
FLAT_TOK = 3739
PARETO_TOK_CAP = int(0.85 * FLAT_TOK)   # MS-23b: 3178
# name -> mode; 2 = top-N windows, "thr" = threshold band
ARMS = [("uwin2", 2), ("uwinthr", "thr")]


def make_windows(n):
    """Registered MS-22 coverage rule, registry order."""
    starts = list(range(0, n - WIN + 1, STEP))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    return [list(range(n))[s:s + WIN] for s in starts]


def pick_windows(wscores, mode):
    if mode == "thr":
        top = max(wscores)
        return [i for i, s in enumerate(wscores) if s >= top - DELTA]
    order = np.argsort(-np.asarray(wscores), kind="stable")
    return sorted(int(i) for i in order[:mode])


def main():
    ap = "--check" in sys.argv
    tools = load_tools()
    tools_by_name = {t["name"]: t for t in tools}
    n = len(tools)
    windows = make_windows(n)
    pools = json.load(open(POOLS))
    svecs = []
    for t in tools:
        us = pools.get(f"{t['server']}__{t['name']}", [])[:8]
        svecs.append(l2(np.stack([l2(embed(u)) for u in us]).mean(0)))
    vecs = np.stack(svecs)
    if ap:
        cov = sorted(set(i for w in windows for i in w))
        print(f"union+windows: {len(windows)} windows size "
              f"{min(len(w) for w in windows)}..{max(len(w) for w in windows)},"
              f" coverage {len(cov)}/{n}; arms={['union'] + [a for a, _ in ARMS]}")
        return 0

    cases = json.load(open(CASES))
    if os.environ.get("MCP_UWIN_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{n} tools, {len(cases)} cases, "
          f"arms={['union'] + [a for a, _ in ARMS]}")
    C = json.load(open(BUDGET))["C"]

    t0 = time.time()
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

    for arm, mode in [("union", None)] + list(ARMS):
        n_done = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            v = l2(embed(c["utterance"]))
            uidx, clusters = shortlist(space, v, 2, None)
            if arm == "union":
                idx = uidx
                route = {"clusters": clusters}
            else:
                wscores = [float((vecs[w] @ v).max()) for w in windows]
                picked = pick_windows(wscores, mode)
                wmem = {i for p in picked for i in windows[p]}
                idx = sorted(set(uidx) | wmem)
                route = {"union_size": len(uidx),
                         "windows_picked": len(picked),
                         "added_by_windows": len(idx) - len(uidx)}
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
        if arm == "union" and not os.environ.get("MCP_UWIN_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO combo "
                      "arms run (protocol MS-23: no verdict on drift). "
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
    if "union" in summary and not os.environ.get("MCP_UWIN_CANARY"):
        u = summary["union"]
        for arm, _ in ARMS:
            if arm not in summary:
                continue
            b = summary[arm]
            neg_ok = int(b["neg"].split("/")[0]) >= SANITY_NEG
            ok_a = (b["acc"] >= u["acc"] + 0.02 and neg_ok)
            print(f"MS-23a {arm}: acc {b['acc']} vs union+0.02 "
                  f"{round(u['acc'] + 0.02, 4)}, neg {b['neg']} -> "
                  f"{'ADOPT AS GATE' if ok_a else 'KEEP UNION'}")
            ok_b = (b["acc"] >= FLAT_ACC
                    and b["mean_prompt_tokens"] <= PARETO_TOK_CAP)
            print(f"MS-23b {arm}: acc {b['acc']} vs flat {FLAT_ACC}, tokens "
                  f"{b['mean_prompt_tokens']} <= {PARETO_TOK_CAP} -> "
                  f"{'NEW PARETO POINT (dominates flat on cost)' if ok_b else 'no flat domination'}")


if __name__ == "__main__":
    sys.exit(main())
