#!/usr/bin/env python
"""MS-21 runner: binary-split walk shortlist ("split to <10", user proposal).

Arms, bars, and sanity gates are registered in docs/PROTOCOL-mcp-selection.md
(MS-21) — committed before any scored run. Imports the frozen run_eval
machinery and run_beam's manifest format verbatim; writes its own
results_btree.jsonl / summary_btree.json. Never edits the frozen files.

    python tools/mcp_probe/run_btree.py --check    # tree shape only, no planner
    python tools/mcp_probe/run_btree.py            # full pass (sanity union first)
    MCP_BTREE_CANARY=1 python tools/mcp_probe/run_btree.py   # 6-case canary
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

RESULTS = os.path.join(HERE, "results_btree.jsonl")
SUMMARY = os.path.join(HERE, "summary_btree.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5                       # registered gate granularity (unchanged)
LEAF = 10                   # registered stop: group smaller than 10
SANITY_ACC = 0.6763         # MS-16a prefixed baseline (deterministic)
SANITY_TOL = 0.005
SANITY_NEG = 20             # of 24
BEAM16_ACC = 0.5971         # MS-20 registered budget bar
BEAM16_TOK = 608
TREE_ARMS = [("bwalk-max", "max", 1), ("bwalk-cent", "cent", 1),
             ("bbeam2-max", "max", 2), ("bbeam2-cent", "cent", 2),
             ("bbeam4-max", "max", 4)]


def build_tree(vecs, members):
    """Balanced binary tree: PC1-median partition, stop below LEAF.
    Left/right naming is SVD-sign-dependent; the partition-as-set is not,
    and every scorer compares both children, so routing is invariant."""
    if len(members) < LEAF:
        return {"members": list(members), "leaf": True}
    X = vecs[members]
    xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    proj = X @ vt[0]
    order = np.argsort(proj, kind="stable")
    mid = len(members) // 2
    left = [members[i] for i in order[:mid]]
    right = [members[i] for i in order[mid:]]
    return {"members": list(members), "leaf": False,
            "left": build_tree(vecs, left),
            "right": build_tree(vecs, right)}


def node_score(node, vecs, v, scorer):
    if scorer == "max":
        return float(np.max(vecs[node["members"]] @ v))
    c = l2(vecs[node["members"]].mean(0))
    return float(c @ v)


def walk(root, vecs, v, scorer):
    """The user's scheme: greedy descend, keep the better-scoring half."""
    node = root
    while not node["leaf"]:
        ls = node_score(node["left"], vecs, v, scorer)
        rs = node_score(node["right"], vecs, v, scorer)
        node = node["left"] if ls >= rs else node["right"]
    return sorted(node["members"])


def beam_walk(root, vecs, v, scorer, width):
    """Beam of width w; path score = MEAN of per-level child scores
    (depth-normalized — the router-walk length-bias lesson, even though
    this balanced tree has equal-depth leaves)."""
    beam = [(0.0, 0, root)]
    while any(not nd["leaf"] for _, _, nd in beam):
        nxt = []
        for s, d, nd in beam:
            if nd["leaf"]:
                nxt.append((s, d, nd))
                continue
            for ch in (nd["left"], nd["right"]):
                nxt.append((s + node_score(ch, vecs, v, scorer), d + 1, ch))
        nxt.sort(key=lambda t: t[0] / max(t[1], 1), reverse=True)
        beam = nxt[:width]
    seen, members = set(), []
    for _, _, nd in beam:
        for m in nd["members"]:
            if m not in seen:
                seen.add(m)
                members.append(m)
    return sorted(members)


def tree_shortlist(root, vecs, v, scorer, width):
    if width == 1:
        return walk(root, vecs, v, scorer)
    return beam_walk(root, vecs, v, scorer, width)


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
    if "union" in summary and not os.environ.get("MCP_BTREE_CANARY"):
        u = summary["union"]
        tree = {k: v for k, v in summary.items() if k != "union"}
        best = max(tree, key=lambda k: tree[k]["acc"]) if tree else None
        if best:
            b = tree[best]
            ok_a = b["acc"] >= u["acc"] and b["hit_rate"] >= 0.878
            print(f"MS-21a replacement: best={best} acc {b['acc']} vs union "
                  f"{u['acc']}, hit {b['hit_rate']} vs 0.878 -> "
                  f"{'GATE REPLACEMENT CANDIDATE' if ok_a else 'KEEP UNION'}")
        bb = summary.get("bbeam2-max")
        if bb:
            ok_b = (bb["acc"] > BEAM16_ACC
                    and bb["mean_prompt_tokens"] <= 1.1 * BEAM16_TOK)
            print(f"MS-21b budget: bbeam2-max acc {bb['acc']} vs beam16 "
                  f"{BEAM16_ACC}, tokens {bb['mean_prompt_tokens']} <= "
                  f"{int(1.1 * BEAM16_TOK)} -> "
                  f"{'BETTER BUDGET SHORTLIST (Pareto note)' if ok_b else 'NO budget win'}")


def main():
    ap = "--check" in sys.argv
    tools = load_tools()
    cases = json.load(open(CASES))
    if ap:
        pools = json.load(open(POOLS))
        svecs = []
        for t in tools:
            us = pools.get(f"{t['server']}__{t['name']}", [])[:8]
            svecs.append(l2(np.stack([l2(embed(u)) for u in us]).mean(0)))
        vecs = np.stack(svecs)
        root = build_tree(vecs, list(range(len(tools))))

        def depths(nd, d=0):
            if nd["leaf"]:
                return [d]
            return depths(nd["left"], d + 1) + depths(nd["right"], d + 1)

        ds = depths(root)
        leaves = []

        def collect(nd):
            if nd["leaf"]:
                leaves.append(len(nd["members"]))
            else:
                collect(nd["left"])
                collect(nd["right"])
        collect(root)
        print(f"tree: depth={min(ds)}..{max(ds)}, leaves={len(leaves)}, "
              f"leaf sizes {min(leaves)}..{max(leaves)}, "
              f"root split {len(root['left']['members'])}/"
              f"{len(root['right']['members'])}")
        return 0

    tools_by_name = {t["name"]: t for t in tools}
    if os.environ.get("MCP_BTREE_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{len(tools)} tools, {len(cases)} cases, "
          f"arms={['union'] + [a for a, _, _ in TREE_ARMS]}")
    C = json.load(open(BUDGET))["C"]
    print(f"MS-2 budget from C_budget.json: C={C}")

    pools = json.load(open(POOLS))
    t0 = time.time()
    svecs = []
    for t in tools:
        us = pools.get(f"{t['server']}__{t['name']}", [])[:8]
        svecs.append(l2(np.stack([l2(embed(u)) for u in us]).mean(0)))
    vecs = np.stack(svecs)
    cents, assign = kmeans(vecs, k=K)
    space = {"cents": cents, "assign": assign, "vecs": vecs}
    root = build_tree(vecs, list(range(len(tools))))
    print(f"spaces built in {time.time()-t0:.0f}s; cluster sizes="
          f"{np.bincount(assign, minlength=K).tolist()}", flush=True)

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

    all_arms = [("union", None, None)] + [
        (a, sc, w) for a, sc, w in TREE_ARMS]
    for arm, scorer, width in all_arms:
        n = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            v = l2(embed(c["utterance"]))
            if arm == "union":
                idx, clusters = shortlist(space, v, 2, None)
                route = {"clusters": clusters, "shortlist": None}
            else:
                idx = tree_shortlist(root, vecs, v, scorer, width)
                route = {"scorer": scorer, "beam": width,
                         "shortlist": None}
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
            n += 1
            if n % 25 == 0:
                el = time.time() - t_start
                print(f"  {arm}: {n} new cases, {el/60:.1f} min elapsed",
                      flush=True)
        print(f"arm {arm} done (+{n} cases)", flush=True)
        if arm == "union" and not os.environ.get("MCP_BTREE_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO tree "
                      "arms run (protocol MS-21: no verdict on drift). "
                      "Fix the baseline first.")
                return 1
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
