#!/usr/bin/env python
"""MS-20 runner: beam-2 path scoring for the gate shortlist.

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-20) — committed before any scored run
(MS-10). This runner imports the frozen run_eval machinery verbatim
(loaders, kmeans, embed, planner, scorer helpers) and never edits it; it
writes its own results_beam.jsonl / summary_beam.json.

    python tools/mcp_probe/run_beam.py            # full pass (all arms)
    MCP_BEAM_CANARY=1 python tools/mcp_probe/run_beam.py   # 6-case canary
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

RESULTS = os.path.join(HERE, "results_beam.jsonl")
SUMMARY = os.path.join(HERE, "summary_beam.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5                      # registered gate granularity (MS-12 optimum)
SANITY_ACC = 0.6763        # MS-16a prefixed baseline (deterministic)
SANITY_TOL = 0.005
SANITY_NEG = 20            # of 24
# name -> (topn_clusters, line_cap); None cap = whole membership
ARMS = [("union", 2, None), ("beam-union", 2, None), ("greedy16", 1, 16),
        ("beam16", 2, 16), ("beam32", 2, 32)]


def manifest_line(t, c_budget):
    args = ", ".join(t["required"]) if t["required"] else "none"
    return f"- {t['server']}__{t['name']}: {t['desc'][:c_budget]}; args: {args}"


def directive_for(tools, c_budget):
    lines = "\n".join(manifest_line(t, c_budget) for t in tools)
    return ("Choose a tool for this request. Available tools:\n" + lines +
            '\nReply with ONLY a JSON object: {"tool": "<name>", "args": {...}} '
            'or {"tool": null} if no tool fits. No other text.')


def shortlist(space, v, topn, cap):
    """MS-20 arms. union: top-2 clusters, members in registry order
    (byte-identical to the production gate / run-2 form). beam arms:
    union ordered by mean(cluster_sim, tool_sim); greedy: top-1 cluster
    ordered by tool sim (cluster term constant there)."""
    sims_c = space["cents"] @ v
    order = np.argsort(-sims_c)[:topn]
    scored = []
    for ci in order:
        csim = float(sims_c[ci])
        for i, a in enumerate(space["assign"]):
            if a == ci:
                scored.append((csim, float(space["vecs"][i] @ v), i))
    if topn == 1:
        scored.sort(key=lambda s: s[1], reverse=True)
    else:
        scored.sort(key=lambda s: 0.5 * (s[0] + s[1]), reverse=True)
    idx = [s[2] for s in scored]
    if cap is None:
        return idx, [int(x) for x in order]
    return idx[:cap], [int(x) for x in order]


def main():
    tools = load_tools()
    tools_by_name = {t["name"]: t for t in tools}
    cases = json.load(open(CASES))
    if os.environ.get("MCP_BEAM_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{len(tools)} tools, {len(cases)} cases, "
          f"arms={[a for a, _, _ in ARMS]}")
    C = json.load(open(BUDGET))["C"]
    print(f"MS-2 budget from C_budget.json: C={C}")

    pools = json.load(open(POOLS))
    t0 = time.time()
    svecs = []
    for t in tools:
        us = pools.get(f"{t['server']}__{t['name']}", [])[:8]
        svecs.append(l2(np.stack([l2(embed(u)) for u in us]).mean(0)))
    cents, assign = kmeans(np.stack(svecs), k=K)
    space = {"cents": cents, "assign": assign, "vecs": np.stack(svecs)}
    print(f"hier-synth k={K} space built in {time.time()-t0:.0f}s; "
          f"cluster sizes={np.bincount(assign, minlength=K).tolist()}",
          flush=True)

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

    for arm, topn, cap in ARMS:
        n = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            v = l2(embed(c["utterance"]))
            idx, clusters = shortlist(space, v, topn, cap)
            short = [tools[i] for i in idx]
            # gold rank in the SHOWN order (1-based; registry order for union)
            stripped = [tools[i]["name"] for i in idx]
            gold_rank = stripped.index(c["want"]) + 1 \
                if c["want"] in stripped else None
            d = directive_for(short, C)
            r = plan(c["utterance"], d)
            rec = {"arm": arm, "case_id": c["id"], "kind": c["kind"],
                   "want": c["want"], **r,
                   "route": {"clusters": clusters,
                             "shortlist": len(short),
                             "gold_rank": gold_rank}}
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
        if arm == "union" and not os.environ.get("MCP_BEAM_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO beam "
                      "arms run (protocol MS-20: no verdict on drift). "
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
    # registered adjudication lines (MS-20 bars)
    if "union" in summary and "beam16" in summary \
            and not os.environ.get("MCP_BEAM_CANARY"):
        u, b16, g16 = summary["union"], summary["beam16"], summary["greedy16"]
        ok_acc = b16["acc"] >= u["acc"]
        ok_tok = b16["mean_prompt_tokens"] < 0.75 * u["mean_prompt_tokens"]
        ok_beam = b16["pos_acc"] >= g16["pos_acc"]
        verdict = "ADOPT beam16" if (ok_acc and ok_tok and ok_beam) \
            else "KEEP UNION"
        print(f"MS-20 primary: beam16 acc {b16['acc']} >= {u['acc']} "
              f"({ok_acc}); tokens {b16['mean_prompt_tokens']} < "
              f"0.75*{u['mean_prompt_tokens']} ({ok_tok}); "
              f"beam16 pos {b16['pos_acc']} >= greedy16 pos "
              f"{g16['pos_acc']} ({ok_beam}) -> {verdict}")
        if not ok_acc and "beam32" in summary:
            b32 = summary["beam32"]
            ok32 = (b32["acc"] >= u["acc"]
                    and b32["mean_prompt_tokens"]
                    < 0.75 * u["mean_prompt_tokens"]
                    and b32["pos_acc"] >= g16["pos_acc"])
            print(f"MS-20 fallback beam32: acc {b32['acc']} vs {u['acc']}, "
                  f"tokens {b32['mean_prompt_tokens']}, pos vs greedy16 "
                  f"{g16['pos_acc']} -> "
                  f"{'ADOPT beam32' if ok32 else 'KEEP UNION'}")


if __name__ == "__main__":
    sys.exit(main())
