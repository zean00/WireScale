#!/usr/bin/env python
"""MS-28 runner: conditional hint (soft cascade).

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-28) — committed before any scored run.
The blind MS-25 intent call runs first; its verdict controls ONLY the
hint: CHAT -> plain union directive (menu intact, no early exit);
TOOL -> union menu plus the registered MS-27 hintfam sentence. Selector
frozen at union top-2 for both branches. Imports frozen machinery
verbatim; writes results_softhint.jsonl / summary_softhint.json.

    python tools/mcp_probe/run_softhint.py --check   # constants + tags
    python tools/mcp_probe/run_softhint.py           # full pass (union first)
    MCP_SOFTHINT_CANARY=1 python tools/mcp_probe/run_softhint.py  # canary
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
from run_step1 import INTENT_DIRECTIVE, intent  # noqa: E402
import run_hint  # noqa: E402
from run_hint import HINT_TAIL, tag, hint_directive  # noqa: E402

RESULTS = os.path.join(HERE, "results_softhint.jsonl")
SUMMARY = os.path.join(HERE, "summary_softhint.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5
SANITY_ACC = 0.6763
SANITY_TOL = 0.005
SANITY_NEG = 20
ARMS = ["softhint"]


def main():
    global tools
    ap = "--check" in sys.argv
    tools = load_tools()
    tools_by_name = {t["name"]: t for t in tools}
    run_hint.tools = tools  # tag() reads its own module global
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
        print(f"softhint arms: {['union'] + ARMS}; intent directive "
              f"{len(INTENT_DIRECTIVE)} chars (frozen MS-25)")
        for u in ("commit changes to my repo", "what is 15% of 240"):
            ci, names, fams = tag(space, l2(embed(u)))
            print(f"  sample {u!r}: tag fams={fams}")
        return 0

    cases = json.load(open(CASES))
    if os.environ.get("MCP_SOFTHINT_CANARY"):
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
                r = plan(c["utterance"], d)
                rec = {"arm": arm, "case_id": c["id"], "kind": c["kind"],
                       "want": c["want"], **r,
                       "route": {**route, "gold_rank": gold_rank}}
            else:
                dec, itok = intent(c["utterance"])
                ci, names, fams = tag(space, v)
                hinted = dec == "TOOL"
                if hinted:
                    hint = ("Automated pre-classification tags this request "
                            "as most resembling the "
                            + ", ".join(fams) + " tool families. "
                            + HINT_TAIL)
                    d = hint_directive(short, C, hint)
                else:
                    d = directive_for(short, C)
                route = {"clusters": clusters, "intent": dec,
                         "hinted": hinted, "intent_tokens": itok,
                         "tag_cluster": ci, "tag_fams": fams}
                r = plan(c["utterance"], d)
                rec = {"arm": arm, "case_id": c["id"], "kind": c["kind"],
                       "want": c["want"], **r,
                       "route": {**route, "gold_rank": gold_rank}}
                rec["prompt_tokens"] = itok + r.get("prompt_tokens", 0)
            if "plan" in rec and rec["plan"] is not None:
                got = rec["plan"]["tool"]
                rec["got_raw"] = got
                if isinstance(got, str):
                    got = got.rsplit("__", 1)[-1]
                rec["got"] = got
                rec["ok"] = (got == c["want"]) if c["want"] is not None \
                    else (got is None)
                if rec["ok"] and c["want"] is not None:
                    rec["schema_ok"] = schema_ok(got,
                                                 rec["plan"].get("args"),
                                                 tools_by_name)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f"  {arm}: {n_done} new cases, {el/60:.1f} min elapsed",
                      flush=True)
        print(f"arm {arm} done (+{n_done} cases)", flush=True)
        if arm == "union" and not os.environ.get("MCP_SOFTHINT_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO softhint "
                      "arm run (protocol MS-28: no verdict on drift). "
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
    if "softhint" in summary and "union" in summary \
            and not os.environ.get("MCP_SOFTHINT_CANARY"):
        u, b = summary["union"], summary["softhint"]
        flags = {"pos_chat": 0, "pos_tool": 0, "neg_chat": 0, "neg_tool": 0,
                 "neg_tool_wrong": 0, "conv": 0}
        okmap = {}
        for line in open(RESULTS):
            r = json.loads(line)
            okmap[(r["arm"], r["case_id"])] = r.get("ok", False)
            if r["arm"] != "softhint":
                continue
            dec = r.get("route", {}).get("intent")
            key = ("pos_chat" if r["kind"] in ("pos", "trap")
                   else "neg_chat") if dec == "CHAT" else \
                  ("pos_tool" if r["kind"] in ("pos", "trap") else "neg_tool")
            flags[key] += 1
            if key == "neg_tool" and not r.get("ok"):
                flags["neg_tool_wrong"] += 1
            if r["kind"] in ("pos", "trap") and dec == "CHAT" \
                    and r.get("ok"):
                flags["conv"] += 1
        gained = sum(1 for (a_, cid), ok in okmap.items() if a_ == "softhint"
                     and ok and not okmap.get(("union", cid), False))
        lost = sum(1 for (a_, cid), ok in okmap.items() if a_ == "softhint"
                   and not ok and okmap.get(("union", cid), False))
        neg_ok = int(b["neg"].split("/")[0]) >= SANITY_NEG
        ok_a = b["acc"] >= u["acc"] + 0.02 and neg_ok
        print(f"intent flags: should-call CHAT {flags['pos_chat']}/254, "
              f"neg CHAT {flags['neg_chat']}/24")
        print(f"MS-28b softhint: negs flagged TOOL {flags['neg_tool']}/24 "
              f"(hint broke {flags['neg_tool_wrong']}); false-CHAT cases "
              f"converted by plain menu {flags['conv']}/{flags['pos_chat']}; "
              f"paired gained {gained} / lost {lost}; pos|hit "
              f"{b['pos_acc_given_hit']} vs {u['pos_acc_given_hit']}")
        print(f"MS-28a softhint: acc {b['acc']} vs union+0.02 "
              f"{round(u['acc'] + 0.02, 4)}, neg {b['neg']} "
              f"(guardrail >=20/24: {neg_ok}) -> "
              f"{'ADOPT CONDITIONAL HINT' if ok_a else 'KEEP UNION'}")


if __name__ == "__main__":
    sys.exit(main())
