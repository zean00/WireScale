#!/usr/bin/env python
"""MS-27 runner: intent as a hint, not a filter.

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-27) — committed before any scored run.
Selector FROZEN at the shipped union (top-2 clusters) for all arms; the
manipulation is ONE inserted sentence telling the planner what the
top-1 cluster classifier guessed (named members or server families),
with a registered "often wrong" clause. Imports frozen machinery
verbatim; writes results_hint.jsonl / summary_hint.json.

    python tools/mcp_probe/run_hint.py --check   # constants + sample tags
    python tools/mcp_probe/run_hint.py           # full pass (union first)
    MCP_HINT_CANARY=1 python tools/mcp_probe/run_hint.py   # 6-case canary
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
from run_beam import directive_for, manifest_line, shortlist  # noqa: E402

RESULTS = os.path.join(HERE, "results_hint.jsonl")
SUMMARY = os.path.join(HERE, "summary_hint.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5
HINT_MEMBERS = 5
SANITY_ACC = 0.6763
SANITY_TOL = 0.005
SANITY_NEG = 20
HINT_TAIL = ("The tag is automated and often wrong — judge from the full "
             "list below.")
ARMS = ["hintname", "hintfam"]


def tag(space, v):
    """Top-1 cluster for the query + its central members / server families."""
    cents, assign, vecs = space["cents"], space["assign"], space["vecs"]
    ci = int(np.argmax(cents @ v))
    members = [i for i, a in enumerate(assign) if a == ci]
    centrality = np.argsort(-(vecs[members] @ cents[ci]), kind="stable")
    names = [f"{tools[i]['server']}__{tools[i]['name']}"
             for i in (members[j] for j in centrality[:HINT_MEMBERS])]
    fam_count = {}
    fam_order = []
    for i in members:
        s = tools[i]["server"]
        if s not in fam_count:
            fam_count[s] = 0
            fam_order.append(s)
        fam_count[s] += 1
    fams = sorted(fam_order, key=lambda s: -fam_count[s])
    return ci, names, fams


def hint_directive(tools, c_budget, hint):
    lines = "\n".join(manifest_line(t, c_budget) for t in tools)
    return ("Choose a tool for this request. " + hint + "\nAvailable tools:\n"
            + lines
            + '\nReply with ONLY a JSON object: {"tool": "<name>", "args": {...}} '
            'or {"tool": null} if no tool fits. No other text.')


def main():
    global tools
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
    cents, assign = kmeans(vecs, k=K)
    space = {"cents": cents, "assign": assign, "vecs": vecs}
    if ap:
        print(f"hint arms: members={HINT_MEMBERS}, arms={['union'] + ARMS}")
        for u in ("commit changes to my repo", "what is 15% of 240"):
            ci, names, fams = tag(space, l2(embed(u)))
            print(f"  sample {u!r}: cluster {ci}, names={names}, fams={fams}")
        return 0

    cases = json.load(open(CASES))
    if os.environ.get("MCP_HINT_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{n} tools, {len(cases)} cases, arms={['union'] + ARMS}")
    C = json.load(open(BUDGET))["C"]
    print("space built", flush=True)

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

    for arm in ["union"] + ARMS:
        n_done = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            v = l2(embed(c["utterance"]))
            idx, clusters = shortlist(space, v, 2, None)
            short = [tools[i] for i in idx]
            stripped = [tools[i]["name"] for i in idx]
            gold_rank = stripped.index(c["want"]) + 1 \
                if c["want"] in stripped else None
            if arm == "union":
                d = directive_for(short, C)
                route = {"clusters": clusters}
            else:
                ci, names, fams = tag(space, v)
                if arm == "hintname":
                    hint = ("Automated pre-classification tags this request "
                            "as most resembling: " + ", ".join(names) + ". "
                            + HINT_TAIL)
                else:
                    hint = ("Automated pre-classification tags this request "
                            "as most resembling the "
                            + ", ".join(fams) + " tool families. "
                            + HINT_TAIL)
                d = hint_directive(short, C, hint)
                route = {"clusters": clusters, "tag_cluster": ci,
                         "tag_names": names, "tag_fams": fams}
            route["shortlist"] = len(short)
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
        if arm == "union" and not os.environ.get("MCP_HINT_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO hint "
                      "arms run (protocol MS-27: no verdict on drift). "
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
    if "union" in summary and not os.environ.get("MCP_HINT_CANARY"):
        u = summary["union"]
        okmap = {}
        for line in open(RESULTS):
            r = json.loads(line)
            okmap[(r["arm"], r["case_id"])] = r.get("ok", False)
        for arm in ARMS:
            if arm not in summary:
                continue
            b = summary[arm]
            gained = sum(1 for c in okmap if c[0] == arm
                         and okmap[c] and not okmap.get(("union", c[1]), False))
            lost = sum(1 for c in okmap if c[0] == arm
                       and not okmap[c] and okmap.get(("union", c[1]), False))
            neg_ok = int(b["neg"].split("/")[0]) >= SANITY_NEG
            ok_a = b["acc"] >= u["acc"] + 0.02 and neg_ok
            print(f"MS-27a {arm}: acc {b['acc']} vs union+0.02 "
                  f"{round(u['acc'] + 0.02, 4)}, neg {b['neg']} "
                  f"(guardrail >=20/24: {neg_ok}) -> "
                  f"{'ADOPT HINT' if ok_a else 'KEEP UNION'}")
            print(f"MS-27b {arm}: pos|hit {b['pos_acc_given_hit']} vs "
                  f"{u['pos_acc_given_hit']}, neg {b['neg']} vs {u['neg']}, "
                  f"paired gained {gained} / lost {lost}")


if __name__ == "__main__":
    sys.exit(main())
