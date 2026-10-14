"""K-series Amendment 3: mid-route scoring (registered pre-data).

Re-runs the four traversal arms on the frozen 60 with persisted
routes; correct iff gold is ANY node on the route. READ/FLAT have
no routes — committed endpoint numbers stand (kgraphqa.json).
kgraphqa.py stays frozen (W precedent: new runner, old untouched).

    python kmid.py --smoke   # 3 questions, never scored
    python kmid.py           # scored run (slot 7 + tunnel)
"""
import argparse
import json
import os
import time

import numpy as np

import chunkrel as CR
import embed_client as EC
import kgraphqa as KG

RES = KG.RES


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arm_walk_routes(q, g, adj, cache, beam):
    """Mirror of KG.arm_walk with the route returned (frozen file kept)."""
    ents = [e for e in g["entities"] if adj[KG.norm_ent(e)]]
    ranked, _ = KG.rank_entries(cache, q, ents)
    top = ranked[:5]
    sysmsg = KG.PICK2_SYS if beam else KG.PICK_SYS
    numbered = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(top))
    r = CR.llm_chat(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content":
          f"Question: {q}\n\nStarting points:\n{numbered}\n\n"
          "Reply with only the number(s)."}],
        temp=0, max_tokens=16)
    usage = r.get("usage", {}).get("prompt_tokens", 0)
    picks = KG.parse_picks(r["choices"][0]["message"]["content"], len(top))
    picks = picks or [1]
    if beam:
        paths = [([KG.norm_ent(top[p - 1])], 1.0 if p == picks[0] else 0.5)
                 for p in picks[:2]]
    else:
        paths = [([KG.norm_ent(top[picks[0] - 1])], 1.0)]
    done = []
    for _hop in range(KG.MAXHOPS):
        cands = []
        for nodes, w in paths:
            node = nodes[-1]
            items = KG.menu_at(node, adj, g)
            if not items:
                done.append((nodes, w))
                continue
            numbered = "\n".join(f"{i + 1}. {lbl}" for i, (lbl, _) in
                                 enumerate(items))
            r = CR.llm_chat(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content":
                  f"Question: {q}\n\nGraph steps at this node:\n{numbered}\n\n"
                  "Reply with only the number(s) (0 to stop here)."}],
                temp=0, max_tokens=16)
            usage += r.get("usage", {}).get("prompt_tokens", 0)
            pk = KG.parse_picks(r["choices"][0]["message"]["content"],
                                len(items))
            if not pk or pk[0] == 0:
                done.append((nodes, w))
                continue
            for j, p in enumerate(pk[:2] if beam else pk[:1]):
                if p > 0:
                    cands.append((nodes + [items[p - 1][1]],
                                  w + (1.0 if j == 0 else 0.5)))
        if beam:
            cands.sort(key=lambda x: -x[1])
            paths = cands[:2]
        else:
            paths = cands[:1]
        if not paths:
            break
    done.extend((n, w) for n, w in paths)
    best = max(done, key=lambda x: x[1]) if done else ([], 0)
    ans = best[0][-1] if best[0] else "UNKNOWN"
    return ans, usage, len(best[0]) - 1, best[0]


def run(limit=0):
    g, adj = KG.load_graph()
    qd = json.load(open(KG.KQFILE))
    if "audit" not in qd:
        raise SystemExit("question file lacks audit key — freeze first")
    with open(os.path.join(RES, "kgraphqa.json")) as f:
        base = json.load(f)
    assert not base.get("smoke")
    qs = qd["eval"][:limit] if limit else qd["eval"]
    canon = {KG.norm_ent(e): e for e in g["entities"]}
    cache, efp = KG.embed_side(g, qs)
    tvecs = [cache[f"{t['s']} {t['r']} {t['o']}"] for t in g["triples"]]
    ents = [e for e in g["entities"] if adj[KG.norm_ent(e)]]

    recs = []
    for i, e in enumerate(qs):
        gold_n = KG.norm_ent(e["gold"])
        wa, wt, _wh, wn = arm_walk_routes(e["q"], g, adj, cache, False)
        ba, bt, _bh, bn = arm_walk_routes(e["q"], g, adj, cache, True)
        la, lt, _lh, ln = KG.arm_look(e["q"], g, adj, canon, cache,
                                      tvecs, ents, beam2=False)
        ca, ct, _ch, cn = KG.arm_look(e["q"], g, adj, canon, cache,
                                      tvecs, ents, beam2=True)
        def cell(ans, nodes, tok):
            return {"end": int(KG.norm_ent(ans) == gold_n),
                    "mid": int(gold_n in nodes),
                    "tok": tok}
        recs.append({"q": e["q"], "gold": e["gold"], "hops": e["hops"],
                     "recipe": e["recipe"],
                     "walk": cell(wa, wn, wt), "beam": cell(ba, bn, bt),
                     "look1": cell(la, ln, lt), "look2": cell(ca, cn, ct)})
        log(f"  q{i + 1}/{len(qs)} h{e['hops']} {e['recipe']:8s} "
            f"walk {recs[-1]['walk']['end']}/{recs[-1]['walk']['mid']} "
            f"beam {recs[-1]['beam']['end']}/{recs[-1]['beam']['mid']} "
            f"look1 {recs[-1]['look1']['end']}/{recs[-1]['look1']['mid']} "
            f"look2 {recs[-1]['look2']['end']}/{recs[-1]['look2']['mid']} "
            "(end/mid)")

    def m(arm, key):
        return float(np.mean([r[arm][key] for r in recs]))

    by = {a: {"mid": m(a, "mid"), "end_fresh": m(a, "end"),
              "end_committed": base["by_arm"][a]["all"] if a in
              base["by_arm"] else None,
              "tok": m(a, "tok")}
          for a in ("walk", "beam", "look1", "look2")}
    sanity = {a: (round(by[a]["end_fresh"] - by[a]["end_committed"], 4)
                  if by[a]["end_committed"] is not None else None)
              for a in by}
    km1d = by["beam"]["mid"] - base["by_arm"]["read"]["all"]
    km1 = "PASS" if km1d >= 0 else "FAIL"
    out = {"tag": "kmid", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "ext_fp": efp,
           "read_committed": base["by_arm"]["read"]["all"],
           "by_arm": by, "endpoint_drift": sanity,
           "verdicts": {"KM1": [km1, round(km1d, 4)],
                        "KM2_descriptive": {
                            a: round(by[a]["mid"] - by[a]["end_fresh"], 4)
                            for a in by}},
           "records": recs}
    name = "kmid_smoke.json" if limit else "kmid.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": by, "endpoint_drift": sanity,
                      "verdicts": out["verdicts"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(3 if a.smoke else 0)
