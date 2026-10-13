#!/usr/bin/env python
"""MS-22 runner: sliding-window shortlist (user proposal).

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-22) — committed before any scored run.
Window = 10 tools, step 7 (overlap 3), full-coverage starts; window score
= max member cosine; pick top-N windows or the threshold band (top-0.03).
Orders: registry (`reg`) and PC1 of the gate's centroid space (`pc1`).
Imports frozen machinery verbatim; writes results_window.jsonl /
summary_window.json.

    python tools/mcp_probe/run_window.py --check   # window shapes only
    python tools/mcp_probe/run_window.py           # full pass (union first)
    MCP_WIN_CANARY=1 python tools/mcp_probe/run_window.py   # 6-case canary
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

RESULTS = os.path.join(HERE, "results_window.jsonl")
SUMMARY = os.path.join(HERE, "summary_window.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5
WIN = 10                    # registered window size
STEP = 7                    # registered step (overlap 3)
DELTA = 0.03                # registered threshold band (descriptive arms)
SANITY_ACC = 0.6763
SANITY_TOL = 0.005
SANITY_NEG = 20
BEAM16_ACC = 0.5971
BEAM16_TOK_CAP = 669       # registered MS-22b budget bar (1.1 x beam16's 608)
# name -> (order, mode); mode: 2/4 = top-N windows, "thr" = threshold band
ARMS = [("win2-reg", "reg", 2), ("win4-reg", "reg", 4),
        ("winthr-reg", "reg", "thr"),
        ("win2-pc1", "pc1", 2), ("win4-pc1", "pc1", 4),
        ("winthr-pc1", "pc1", "thr")]


def make_windows(order, n):
    """Registered coverage rule: starts = range(0, n-9, STEP); append n-WIN
    if the last start misses it. Every tool covered, windows WIN-sized."""
    starts = list(range(0, n - WIN + 1, STEP))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    return [sorted(order[s:s + WIN]) for s in starts]


def pc1_order(vecs):
    xc = vecs - vecs.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    proj = vecs @ vt[0]
    return [int(i) for i in np.argsort(proj, kind="stable")]


def pick_windows(windows, wscores, mode):
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
    pools = json.load(open(POOLS))
    svecs = []
    for t in tools:
        us = pools.get(f"{t['server']}__{t['name']}", [])[:8]
        svecs.append(l2(np.stack([l2(embed(u)) for u in us]).mean(0)))
    vecs = np.stack(svecs)
    orders = {"reg": list(range(n)), "pc1": pc1_order(vecs)}
    wins = {o: make_windows(w, n) for o, w in orders.items()}
    if ap:
        for o in ("reg", "pc1"):
            sizes = [len(w) for w in wins[o]]
            cov = sorted(set(i for w in wins[o] for i in w))
            print(f"{o}: {len(wins[o])} windows, sizes {min(sizes)}.."
                  f"{max(sizes)}, coverage {len(cov)}/{n}, "
                  f"first-window {[tools[i]['name'] for i in wins[o][0]][:3]}")
        return 0

    cases = json.load(open(CASES))
    if os.environ.get("MCP_WIN_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{n} tools, {len(cases)} cases, arms={['union'] + [a for a, _, _ in ARMS]}")
    C = json.load(open(BUDGET))["C"]

    t0 = time.time()
    cents, assign = kmeans(vecs, k=K)
    space = {"cents": cents, "assign": assign, "vecs": vecs}
    print(f"spaces built in {time.time()-t0:.0f}s", flush=True)

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

    all_arms = [("union", None, None)] + list(ARMS)
    for arm, order_name, mode in all_arms:
        n_done = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            v = l2(embed(c["utterance"]))
            if arm == "union":
                idx, clusters = shortlist(space, v, 2, None)
                route = {"clusters": clusters}
            else:
                windows = wins[order_name]
                wscores = [float(np.max(vecs[w] @ v)) for w in windows]
                picked = pick_windows(windows, wscores, mode)
                members = sorted({i for p in picked for i in windows[p]})
                idx = members
                route = {"order": order_name, "windows": len(windows),
                         "picked": len(picked),
                         "picked_scores": [round(wscores[p], 4)
                                           for p in picked]}
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
        if arm == "union" and not os.environ.get("MCP_WIN_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO window "
                      "arms run (protocol MS-22: no verdict on drift). "
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
    if "union" in summary and not os.environ.get("MCP_WIN_CANARY"):
        u = summary["union"]
        win_arms = {k: v for k, v in summary.items()
                    if k.startswith("win") and not k.startswith("winthr")}
        best = max(win_arms, key=lambda k: win_arms[k]["acc"]) \
            if win_arms else None
        if best:
            b = win_arms[best]
            ok_a = b["acc"] >= u["acc"] and b["hit_rate"] >= 0.878
            print(f"MS-22a replacement: best={best} acc {b['acc']} vs union "
                  f"{u['acc']}, hit {b['hit_rate']} vs 0.878 -> "
                  f"{'GATE REPLACEMENT CANDIDATE' if ok_a else 'KEEP UNION'}")
        w2 = summary.get("win2-pc1")
        if w2:
            ok_b = (w2["acc"] > BEAM16_ACC
                    and w2["mean_prompt_tokens"] <= BEAM16_TOK_CAP)
            print(f"MS-22b budget: win2-pc1 acc {w2['acc']} vs beam16 "
                  f"{BEAM16_ACC}, tokens {w2['mean_prompt_tokens']} <= "
                  f"{BEAM16_TOK_CAP} -> "
                  f"{'BETTER BUDGET SHORTLIST (Pareto note)' if ok_b else 'NO budget win'}")


if __name__ == "__main__":
    sys.exit(main())
