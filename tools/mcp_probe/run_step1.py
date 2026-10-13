#!/usr/bin/env python
"""MS-25 runner: two-step gate — blind intent check, then winner band.

Arms, bars, and sanity gates are registered in
docs/PROTOCOL-mcp-selection.md (MS-25) — committed before any scored run.
Step 1: manifest-blind LLM call answers TOOL or CHAT (temp 0, 4 tokens).
CHAT -> final {"tool": null} without a planner call. TOOL -> band08
shortlist (delta/cap from the committed run_band) -> planner. Sanity
`union` anchor reruns first. Imports frozen machinery verbatim; writes
results_step1.jsonl / summary_step1.json.

    python tools/mcp_probe/run_step1.py --check   # constants only
    python tools/mcp_probe/run_step1.py           # full pass (union first)
    MCP_STEP1_CANARY=1 python tools/mcp_probe/run_step1.py  # 6-case canary
"""
import json
import os
import sys
import time

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "..", "web")
sys.path.insert(0, WEB)
sys.path.insert(0, HERE)
os.chdir(WEB)
import server  # noqa: E402
from run_eval import embed, kmeans, load_tools, l2, plan, schema_ok  # noqa: E402
from run_beam import directive_for, shortlist  # noqa: E402
from run_band import DELTA, CAP, band_indices  # noqa: E402

RESULTS = os.path.join(HERE, "results_step1.jsonl")
SUMMARY = os.path.join(HERE, "summary_step1.json")
BUDGET = os.path.join(HERE, "C_budget.json")
POOLS = os.path.join(HERE, "synth_pools.json")
CASES = os.path.join(HERE, "eval_cases.json")

K = 5
SANITY_ACC = 0.6763
SANITY_TOL = 0.005
SANITY_NEG = 20
UNION_TOK = 2265
BUDGET_TOK_CAP = int(0.70 * UNION_TOK)   # MS-25b: 1585
INTENT_DIRECTIVE = (
    "Decide what kind of message this is. Reply with exactly one word:\n"
    "TOOL - the assistant should call one of its tools (files, git, time, "
    "smart home, web, or similar) to fulfill this message;\n"
    "CHAT - plain conversation that needs no tool.\n"
    "One word only: TOOL or CHAT.")


def intent(utterance):
    """Blind step-1 call: no manifest, no tool names. Returns
    (decision, prompt_tokens)."""
    p = server.build_prompt([], utterance, INTENT_DIRECTIVE)
    r = requests.post(f"{server.LLAMA}/v1/completions", json={
        "prompt": p, "max_tokens": 4, "temperature": 0,
        "cache_prompt": True, "id_slot": server.SLOT}, timeout=300)
    r.raise_for_status()
    text = r.json()["choices"][0]["text"].strip().upper()
    dec = "TOOL" if text.startswith("TOOL") else "CHAT"
    return dec, r.json()["usage"]["prompt_tokens"]


def main():
    ap = "--check" in sys.argv
    tools = load_tools()
    tools_by_name = {t["name"]: t for t in tools}
    n = len(tools)
    if ap:
        print(f"two-step gate: blind intent call -> band08 "
              f"(delta={DELTA}, cap={CAP}); {n} tools; "
              f"arms=['union', 'step1-band08']; budget cap {BUDGET_TOK_CAP} tok")
        return 0

    cases = json.load(open(CASES))
    if os.environ.get("MCP_STEP1_CANARY"):
        pos = [c for c in cases if c["kind"] == "pos"][:2]
        neg = [c for c in cases if c["kind"] == "neg"][:2]
        trap = [c for c in cases if c["kind"] == "trap"][:2]
        cases = pos + neg + trap
        print(f"CANARY subset: {[c['id'] for c in cases]}")
    print(f"{n} tools, {len(cases)} cases, arms=['union', 'step1-band08']")
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

    for arm in ("union", "step1-band08"):
        n_done = 0
        for c in cases:
            if (arm, c["id"]) in done:
                continue
            if arm == "union":
                v = l2(embed(c["utterance"]))
                idx, clusters = shortlist(space, v, 2, None)
                short = [tools[i] for i in idx]
                route = {"clusters": clusters, "shortlist": len(short)}
                d = directive_for(short, C)
                r = plan(c["utterance"], d)
                rec = {"arm": arm, "case_id": c["id"], "kind": c["kind"],
                       "want": c["want"], **r, "route": route}
            else:
                dec, ipt = intent(c["utterance"])
                rec = {"arm": arm, "case_id": c["id"], "kind": c["kind"],
                       "want": c["want"], "intent": dec, "intent_pt": ipt}
                if dec == "CHAT":
                    # planner never called; final answer is null
                    rec.update({"plan": None, "prompt_tokens": ipt,
                                "ms": None, "got": None, "ok":
                                c["want"] is None,
                                "route": {"shortlist": 0,
                                          "intent_exit": True}})
                    if c["want"] is not None:
                        rec["rejected_by_intent"] = True
                else:
                    v = l2(embed(c["utterance"]))
                    sims = vecs @ v
                    idx, band_size, capped = band_indices(sims, CAP)
                    short = [tools[i] for i in idx]
                    stripped = [tools[i]["name"] for i in idx]
                    gold_rank = stripped.index(c["want"]) + 1 \
                        if c["want"] in stripped else None
                    d = directive_for(short, C)
                    r = plan(c["utterance"], d)
                    rec.update({**r,
                                "route": {"band_size": int(band_size),
                                          "capped": capped,
                                          "shortlist": len(short),
                                          "gold_rank": gold_rank}})
                    if "plan" in r and r["plan"] is not None:
                        got = r["plan"]["tool"]
                        rec["got_raw"] = got
                        if isinstance(got, str):
                            got = got.rsplit("__", 1)[-1]
                        rec["got"] = got
                    rec["prompt_tokens"] = ipt + r.get("prompt_tokens", 0)
            if rec.get("plan") is not None:
                got = rec["plan"]["tool"]
                if isinstance(got, str):
                    got = got.rsplit("__", 1)[-1]
                rec["got"] = got
                rec["ok"] = (got == c["want"]) if c["want"] is not None \
                    else (got is None)
                if rec["ok"] and c["want"] is not None:
                    rec["schema_ok"] = schema_ok(
                        got, rec["plan"].get("args"), tools_by_name)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f"  {arm}: {n_done} new cases, {el/60:.1f} min elapsed",
                      flush=True)
        print(f"arm {arm} done (+{n_done} cases)", flush=True)
        if arm == "union" and not os.environ.get("MCP_STEP1_CANARY"):
            u = summarize_arm("union")
            print(f"sanity union: acc={u['acc']} (registered 0.6763 "
                  f"±{SANITY_TOL}), neg={u['neg']}")
            if abs(u["acc"] - SANITY_ACC) > SANITY_TOL or u["neg"] < SANITY_NEG:
                print("SANITY FAIL — prefixed baseline drifted; NO step1 "
                      "arm run (protocol MS-25: no verdict on drift). "
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
        if r.get("ms") is not None:
            a["ms"].append(r["ms"])
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
                                       "phit": [0, 0], "chat_exit": [0, 0]})
        if "error" in r:
            continue
        a["n"] += 1
        a["ok"] += int(r.get("ok", False))
        a["pt"].append(r.get("prompt_tokens", 0))
        if r.get("ms") is not None:
            a["ms"].append(r["ms"])
        if r.get("ok") and "schema_ok" in r:
            a["schema"].append(int(r["schema_ok"]))
        hit = r.get("route", {}).get("gold_rank") is not None
        if r.get("intent") == "CHAT":
            a["chat_exit"][1] += 1
            a["chat_exit"][0] += int(r["kind"] == "neg")
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
            "chat_exits": f"{a['chat_exit'][0]}/{a['chat_exit'][1]}",
            "mean_prompt_tokens": round(sum(a["pt"]) / max(len(a["pt"]), 1), 1),
            "mean_ms": round(sum(a["ms"]) / max(len(a["ms"]), 1), 1)}
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if "union" in summary and not os.environ.get("MCP_STEP1_CANARY"):
        u = summary["union"]
        if "step1-band08" in summary:
            b = summary["step1-band08"]
            neg_ok = int(b["neg"].split("/")[0]) >= SANITY_NEG
            ok_a = b["acc"] >= u["acc"] + 0.02 and neg_ok
            print(f"MS-25a step1-band08: acc {b['acc']} vs union+0.02 "
                  f"{round(u['acc'] + 0.02, 4)}, neg {b['neg']} -> "
                  f"{'ADOPT AS GATE' if ok_a else 'KEEP UNION'}")
            ok_b = (b["acc"] >= u["acc"] - 0.01
                    and b["mean_prompt_tokens"] <= BUDGET_TOK_CAP)
            print(f"MS-25b step1-band08: acc {b['acc']} vs union-0.01 "
                  f"{round(u['acc'] - 0.01, 4)}, tokens "
                  f"{b['mean_prompt_tokens']} <= {BUDGET_TOK_CAP} -> "
                  f"{'BUDGET PARETO (gate accuracy at ~half cost)' if ok_b else 'no budget win'}")


if __name__ == "__main__":
    sys.exit(main())
