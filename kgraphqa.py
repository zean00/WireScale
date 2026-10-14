"""K-series: decision layers on knowledge-graph QA (protocol.md, registered).

Builds a knowledge graph from the pinned corpus (frozen model, temp 0),
generates path-grounded questions (1-hop and 2-hop, direct/indirect),
then scores four arms at their natural budgets:

  FLAT  one call over the whole graph (anchor)
  READ  embedder subgraph (2 hops from top-3 entries, top-30 triples)
  WALK  hop-by-hop menu traversal, terminal node = answer
  BEAM  traversal keeping a frontier of 2 (the replicated lab win)

    python kgraphqa.py --smoke   # tiny self-contained check, never scored
    python kgraphqa.py --build   # extract triples -> results/kgraph.json
    python kgraphqa.py --gen     # path-grounded questions -> k_questions.json
    python kgraphqa.py           # scored run (needs graph, questions, tunnel)
"""
import argparse
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np

import chunkrel as CR
import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GFILE = os.path.join(RES, "kgraph.json")
KQFILE = os.path.join(RES, "k_questions.json")
K_SEED, N_Q, MAXHOPS = 20261014, 60, 3
READ_ENTRIES, READ_BREADTH, READ_TOP = 3, 12, 30
WALK_MENU_MAX, BEAM_GUARD, K1_GUARD = 20, 2.2, 1.1
# LOOKAHEAD (Amendment 2, registered caps)
LOOK_L1_CAP, LOOK_EXT_CAP = 24, 12
LOOK_POOL2, LOOK_POOL3, LOOK_MENU = 300, 800, 60
LOOK_SEGS, LOOK_PATH_HOPS = 2, 3

EXTRACT_SYS = ("You extract knowledge-graph triples from text. Reply with "
               "ONLY a JSON array like [{\"s\":\"...\",\"r\":\"...\","
               "\"o\":\"...\"}]. s and o are short noun phrases (max 4 "
               "words) naming things/concepts; r is a short verb phrase "
               "(max 3 words, lowercase). Up to 6 triples. Only facts the "
               "text states explicitly. No commentary.")
QGEN_SYS = "Output only the question itself. No preamble, no answer, no quotes."
QRECIPES = {
    "direct": ("Write one question of at most 20 words whose answer is "
               "exactly: {gold}. Reuse the vocabulary of the facts where "
               "natural. The question must be answerable from these facts "
               "alone.\n\nFACTS:\n{facts}\n\nSOURCE:\n{src}"),
    "indirect": ("Write one question of at most 25 words that a person with "
                 "a concrete need would ask when these facts are exactly "
                 "what they need. Do NOT reuse the wording of the answer "
                 "entity or of the relations; describe the underlying need "
                 "or goal.\n\nFACTS:\n{facts}\n\nSOURCE:\n{src}"),
}
ANSWER_SYS = ("You answer a question using ONLY the given facts. Reply with "
              "ONLY the exact entity name from the facts that answers the "
              "question, or UNKNOWN.")
PICK_SYS = ("You pick the graph step that best helps answer a question. "
            "Reply with ONLY the number of your choice, or 0 to stop.")
PICK2_SYS = ("You pick the two graph steps that best help answer a "
             "question, best first. Reply with ONLY two numbers separated "
             "by a comma, best first.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_ent(x):
    x = re.sub(r"\s+", " ", str(x).strip().strip('"\').,;:!?').lower())
    for a in ("the ", "a ", "an "):
        if x.startswith(a):
            x = x[len(a):]
    return x.strip()


# ------------------------------------------------------------------ build

def extract_triples(cm):
    raw, i = [], 0
    for doc, ch in cm.items():
        for ci, chunk in enumerate(ch):
            r = CR.llm_chat(
                [{"role": "system", "content": EXTRACT_SYS},
                 {"role": "user", "content": chunk}],
                temp=0, max_tokens=500, seed=K_SEED + i)
            i += 1
            txt = r["choices"][0]["message"]["content"]
            m = re.search(r"\[.*\]", txt, re.S)
            if not m:
                continue
            try:
                arr = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            for t in arr if isinstance(arr, list) else []:
                s, rr, o = (str(t.get(k, "")).strip() for k in ("s", "r", "o"))
                if not s or not rr or not o:
                    continue
                if max(len(s.split()), len(o.split())) > 6 \
                        or len(rr.split()) > 4:
                    continue
                raw.append({"s": s, "r": rr, "o": o, "doc": doc,
                            "chunk": ci})
            if i % 20 == 0:
                log(f"  {i} chunks, {len(raw)} raw triples")
    return raw


def build_graph(cm):
    raw = extract_triples(cm)
    case = defaultdict(Counter)
    for t in raw:
        for k in ("s", "o"):
            case[norm_ent(t[k])][t[k]] += 1
    canon = {k: c.most_common(1)[0][0] for k, c in case.items()}
    seen, triples = set(), []
    for t in raw:
        key = (norm_ent(t["s"]), t["r"].lower(), norm_ent(t["o"]))
        if key in seen:
            continue
        seen.add(key)
        triples.append({"s": canon[key[0]], "r": t["r"].lower(),
                        "o": canon[key[2]], "doc": t["doc"],
                        "chunk": t["chunk"]})
    adj = defaultdict(list)
    for ti, t in enumerate(triples):
        adj[norm_ent(t["s"])].append(ti)
        adj[norm_ent(t["o"])].append(ti)
    g = {"commit": CR.git_commit(), "model_fp": EC.fingerprint(),
         "n_chunks": sum(len(ch) for ch in cm.values()),
         "entities": sorted(canon.values()),
         "n_entities": len(canon), "n_triples": len(triples),
         "triples": triples,
         "degree": {k: len(v) for k, v in adj.items()}}
    with open(GFILE, "w") as f:
        json.dump(g, f, indent=1, sort_keys=True)
    log(f"wrote {GFILE}: {g['n_entities']} entities, "
        f"{g['n_triples']} triples from {g['n_chunks']} chunks")
    return g


def load_graph():
    with open(GFILE) as f:
        g = json.load(f)
    adj = defaultdict(list)
    for ti, t in enumerate(g["triples"]):
        adj[norm_ent(t["s"])].append(ti)
        adj[norm_ent(t["o"])].append(ti)
    return g, adj


# ------------------------------------------------------------------ gen

def sample_paths(g, adj, rng, n_one, n_two):
    ents = [e for e in g["entities"] if adj[norm_ent(e)]]
    one, two, seen = [], [], set()
    guard = 0
    while (len(one) < n_one or len(two) < n_two) and guard < 6000:
        guard += 1
        a = norm_ent(ents[int(rng.integers(len(ents)))])
        ta = adj[a][int(rng.integers(len(adj[a])))]
        t = g["triples"][ta]
        b = norm_ent(t["o"]) if norm_ent(t["s"]) == a else norm_ent(t["s"])
        if len(one) < n_one and (ta, None) not in seen:
            one.append(([t], t["o"]))
            seen.add((ta, None))
            continue
        if len(two) < n_two and adj[b]:
            tb = adj[b][int(rng.integers(len(adj[b])))]
            t2 = g["triples"][tb]
            c = (norm_ent(t2["o"]) if norm_ent(t2["s"]) == b
                 else norm_ent(t2["s"]))
            if c not in (a, b) and (ta, tb) not in seen:
                two.append(([t, t2],
                            t2["o"] if norm_ent(t2["s"]) == b else t2["s"]))
                seen.add((ta, tb))
    return one, two


def gen_questions():
    g, adj = load_graph()
    rng = np.random.default_rng(K_SEED)
    one, two = sample_paths(g, adj, rng, 30, 30)
    log(f"paths: 1-hop {len(one)}, 2-hop {len(two)}")
    cm = CR.build_chunkmap()
    out, i = [], 0
    for hops, paths in ((1, one), (2, two)):
        for path, gold in paths:
            facts = "\n".join(f"- {t['s']} — {t['r']} — {t['o']}"
                              for t in path)
            src = "\n..\n".join(cm[t["doc"]][t["chunk"]][:300] for t in path)
            for recipe, tmpl in QRECIPES.items():
                r = CR.llm_chat(
                    [{"role": "system", "content": QGEN_SYS},
                     {"role": "user", "content":
                      tmpl.format(gold=gold, facts=facts, src=src)}],
                    temp=0.8, max_tokens=64,
                    seed=K_SEED * 100003 + 13 * i)
                i += 1
                q = r["choices"][0]["message"]["content"].strip().strip('"')
                out.append({"q": q, "gold": gold, "hops": hops,
                            "recipe": recipe,
                            "path": [[t["s"], t["r"], t["o"],
                                      t["doc"], t["chunk"]] for t in path]})
        log(f"  {hops}-hop questions done ({i} generations)")
    order = np.random.default_rng(K_SEED + 1).permutation(len(out))
    data = {"eval": [out[j] for j in order],
            "note": ("K-series eval, path-grounded; UNAUDITED until the "
                     "audit key is added (R-series policy)") }
    with open(KQFILE, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    log(f"wrote {KQFILE} — {len(out)} questions (pre-audit)")


# ------------------------------------------------------------------ embed

def embed_side(g, qs):
    import extembed as EE
    fp, dim = EE.ext_fingerprint()
    cache = EE.load_ext_cache(fp)
    texts = sorted(set(g["entities"])
                   | {f"{t['s']} {t['r']} {t['o']}" for t in g["triples"]}
                   | {e["q"] for e in qs})
    EE.ext_embed(texts, cache, fp)
    return cache, fp


def rank_entries(cache, q, ents):
    qv = cache[q]
    sims = np.asarray([float(cache[e] @ qv) for e in ents])
    return [ents[i] for i in np.argsort(-sims)], sims


# ------------------------------------------------------------------ arms

def triples_prompt(triples):
    return "\n".join(f"- {t['s']} — {t['r']} — {t['o']}" for t in triples)


def ask_answer(q, block):
    r = CR.llm_chat(
        [{"role": "system", "content": ANSWER_SYS},
         {"role": "user", "content":
          f"Facts:\n{block}\n\nQuestion: {q}\n\nAnswer entity:"}],
        temp=0, max_tokens=32)
    return (r["choices"][0]["message"]["content"].strip(),
            r.get("usage", {}))


def parse_picks(txt, k):
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    return [n for n in nums if 0 <= n <= k][:2]


def menu_at(node_norm, adj, g):
    tis = adj[node_norm][:WALK_MENU_MAX]
    items = []
    for ti in tis:
        t = g["triples"][ti]
        other = norm_ent(t["o"]) if norm_ent(t["s"]) == node_norm \
            else norm_ent(t["s"])
        items.append((f"{t['s']} — {t['r']} — {t['o']}", other))
    return items


def arm_flat(q, g):
    """FLAT-2HALF (amended, protocol.md): whole graph exceeds the
    production slot's 8192-token ceiling, so read two deterministic
    index-halves and tie-break conflicts. No selection, no ranking."""
    tris = g["triples"]
    mid = (len(tris) + 1) // 2
    a1, u1 = ask_answer(q, triples_prompt(tris[:mid]))
    a2, u2 = ask_answer(q, triples_prompt(tris[mid:]))
    tok = u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0)
    if a1 == a2 or a2.upper() == "UNKNOWN":
        return a1, tok
    if a1.upper() == "UNKNOWN":
        return a2, tok
    r = CR.llm_chat(
        [{"role": "system", "content": ANSWER_SYS},
         {"role": "user", "content":
          f"Question: {q}\n\nCandidate answers:\nA. {a1}\nB. {a2}\n\n"
          "Reply with only the better candidate (A or B)."}],
        temp=0, max_tokens=8)
    pick = r["choices"][0]["message"]["content"].strip().upper()[:1]
    return (a1 if pick == "A" else a2), tok + \
        r.get("usage", {}).get("prompt_tokens", 0)


def arm_read(q, g, adj, cache):
    ents = [e for e in g["entities"] if adj[norm_ent(e)]]
    ranked, _ = rank_entries(cache, q, ents)
    frontier, seen_n, tis = [norm_ent(ranked[0])], set(), set()
    for _hop in range(2):
        nxt = []
        for n in frontier:
            if n in seen_n:
                continue
            seen_n.add(n)
            cand = adj[n]
            if len(cand) > READ_BREADTH:
                cand = sorted(cand, key=lambda ti: -float(
                    cache[f"{g['triples'][ti]['s']} {g['triples'][ti]['r']} "
                          f"{g['triples'][ti]['o']}"] @ cache[q]))[:READ_BREADTH]
            for ti in cand:
                tis.add(ti)
                t = g["triples"][ti]
                other = norm_ent(t["o"]) if norm_ent(t["s"]) == n \
                    else norm_ent(t["s"])
                if other not in seen_n:
                    nxt.append(other)
        frontier = nxt[:READ_ENTRIES * READ_BREADTH]
    tris = [g["triples"][ti] for ti in tis]
    tris.sort(key=lambda t: -float(
        cache[f"{t['s']} {t['r']} {t['o']}"] @ cache[q]))
    ans, u = ask_answer(q, triples_prompt(tris[:READ_TOP]))
    return ans, u.get("prompt_tokens", 0)


def arm_walk(q, g, adj, cache, beam=False):
    ents = [e for e in g["entities"] if adj[norm_ent(e)]]
    ranked, _ = rank_entries(cache, q, ents)
    top = ranked[:5]
    numbered = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(top))
    sysmsg = PICK2_SYS if beam else PICK_SYS
    r = CR.llm_chat(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content":
          f"Question: {q}\n\nStarting points:\n{numbered}\n\n"
          "Reply with only the number(s)."}],
        temp=0, max_tokens=16)
    usage = r.get("usage", {}).get("prompt_tokens", 0)
    picks = parse_picks(r["choices"][0]["message"]["content"], len(top))
    picks = picks or [1]
    if beam:
        paths = [([norm_ent(top[p - 1])], 1.0 if p == picks[0] else 0.5)
                 for p in picks[:2]]
    else:
        paths = [([norm_ent(top[picks[0] - 1])], 1.0)]
    done = []
    for _hop in range(MAXHOPS):
        cands = []
        for nodes, w in paths:
            node = nodes[-1]
            items = menu_at(node, adj, g)
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
            pk = parse_picks(r["choices"][0]["message"]["content"],
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
    # canonical casing for readability of records
    return ans, usage, len(best[0]) - 1


# ------------------------------------------------------------- lookahead

def far_node(g, ti, from_norm):
    t = g["triples"][ti]
    return norm_ent(t["o"]) if norm_ent(t["s"]) == from_norm \
        else norm_ent(t["s"])


def enumerate_paths(g, adj, tvecs, anchor, qv):
    """Simple paths up to LOOK_PATH_HOPS from anchor, registered caps.
    Ranking is embedder cosine of TRIPLE texts (cached) — no new embeds."""
    def score(tis):
        return float(np.mean([tvecs[ti] @ qv for ti in tis]))
    l1 = sorted(adj[anchor],
                key=lambda ti: -float(tvecs[ti] @ qv))[:LOOK_L1_CAP]
    cur = [{"nodes": [anchor, far_node(g, ti, anchor)], "tis": [ti]}
           for ti in l1]
    allp = list(cur)
    for pool in (LOOK_POOL2, LOOK_POOL3):
        nxt = []
        for p in cur:
            end = p["nodes"][-1]
            for ti in adj[end]:
                if ti in p["tis"]:
                    continue
                fn = far_node(g, ti, end)
                if fn in p["nodes"]:
                    continue
                nxt.append({"nodes": p["nodes"] + [fn],
                            "tis": p["tis"] + [ti]})
        nxt.sort(key=lambda p: -score(p["tis"]))
        cur = nxt[:pool]
        allp += cur
    allp.sort(key=lambda p: -score(p["tis"]))
    return allp[:LOOK_MENU]


def path_label(g, p, canon):
    parts = [canon.get(p["nodes"][0], p["nodes"][0])]
    for i, ti in enumerate(p["tis"]):
        parts.append(f"-{g['triples'][ti]['r']}-")
        parts.append(canon.get(p["nodes"][i + 1], p["nodes"][i + 1]))
    return " ".join(parts)


def arm_look(q, g, adj, canon, cache, tvecs, ents, beam2):
    usage = 0
    ranked, _ = rank_entries(cache, q, ents)
    top5 = ranked[:5]
    sysmsg = PICK2_SYS if beam2 else PICK_SYS
    numbered = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(top5))
    r = CR.llm_chat(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content":
          f"Question: {q}\n\nStarting points:\n{numbered}\n\n"
          "Reply with only the number(s)."}],
        temp=0, max_tokens=16)
    usage += r.get("usage", {}).get("prompt_tokens", 0)
    picks = parse_picks(r["choices"][0]["message"]["content"], len(top5))
    picks = picks or [1]
    frontiers = [{"nodes": [norm_ent(top5[p - 1])],
                  "w": 1.0 if j == 0 else 0.5}
                 for j, p in enumerate(picks[:2] if beam2 else picks[:1])]
    done = []
    for _seg in range(LOOK_SEGS):
        nxt = []
        for f in frontiers:
            anchor = f["nodes"][-1]
            paths = enumerate_paths(g, adj, tvecs, anchor, cache[q])
            if not paths:
                done.append(f)
                continue
            labels = [path_label(g, p, canon) for p in paths]
            numbered = "\n".join(f"{i + 1}. {l}"
                                 for i, l in enumerate(labels))
            r = CR.llm_chat(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content":
                  f"Question: {q}\n\nRoutes from here "
                  f"({len(labels)}):\n{numbered}\n\n"
                  "Reply with only the number(s) (0 to stop here)."}],
                temp=0, max_tokens=16)
            usage += r.get("usage", {}).get("prompt_tokens", 0)
            pk = parse_picks(r["choices"][0]["message"]["content"],
                             len(paths))
            if not pk or pk[0] == 0:
                done.append(f)
                continue
            for j, p in enumerate(pk[:2] if beam2 else pk[:1]):
                if p > 0:
                    nxt.append({"nodes": f["nodes"]
                                + paths[p - 1]["nodes"][1:],
                                "w": f["w"] + (1.0 if j == 0 else 0.5)})
        if beam2:
            nxt.sort(key=lambda x: -x["w"])
            frontiers = nxt[:2]
        else:
            frontiers = nxt[:1]
        if not frontiers:
            break
    done.extend(frontiers)
    best = max(done, key=lambda x: x["w"]) if done \
        else {"nodes": ["UNKNOWN"], "w": 0.0}
    return best["nodes"][-1], usage, len(best["nodes"]) - 1, best["nodes"]


def run_look(limit=0):
    g, adj = load_graph()
    with open(KQFILE) as f:
        qd = json.load(f)
    if "audit" not in qd:
        raise SystemExit("question file lacks audit key — freeze first")
    with open(os.path.join(RES, "kgraphqa.json")) as f:
        base = json.load(f)
    assert not base.get("smoke"), "baseline file is a smoke artifact"
    qs = qd["eval"][:limit] if limit else qd["eval"]
    canon = {norm_ent(e): e for e in g["entities"]}
    cache, efp = embed_side(g, qs)
    tvecs = [cache[f"{t['s']} {t['r']} {t['o']}"] for t in g["triples"]]
    ents = [e for e in g["entities"] if adj[norm_ent(e)]]
    log(f"lookahead: {len(qs)} questions, graph {g['n_triples']} triples")

    recs = []
    for i, e in enumerate(qs):
        gold_n = norm_ent(e["gold"])
        a1, t1, h1, n1 = arm_look(e["q"], g, adj, canon, cache, tvecs,
                                  ents, beam2=False)
        a2, t2, h2, n2 = arm_look(e["q"], g, adj, canon, cache, tvecs,
                                  ents, beam2=True)
        recs.append({
            "q": e["q"], "gold": e["gold"], "hops": e["hops"],
            "recipe": e["recipe"],
            "look1": [int(norm_ent(a1) == gold_n), a1, t1, h1],
            "look2": [int(norm_ent(a2) == gold_n), a2, t2, h2],
            "gold_on_route2": int(gold_n in n2[:-1])})
        log(f"  q{i + 1}/{len(qs)} h{e['hops']} {e['recipe']:8s} "
            f"look1={recs[-1]['look1'][0]} look2={recs[-1]['look2'][0]} "
            f"onroute={recs[-1]['gold_on_route2']}")

    def acc(arm, key=None, val=None):
        v = [r[arm][0] for r in recs if key is None or r[key] == val]
        return float(np.mean(v)) if v else None

    def toks(arm):
        return float(np.mean([r[arm][2] for r in recs]))

    by = {a: {"all": acc(a), "onehop": acc(a, "hops", 1),
              "twohop": acc(a, "hops", 2),
              "direct": acc(a, "recipe", "direct"),
              "indirect": acc(a, "recipe", "indirect"),
              "tok": toks(a)}
          for a in ("look1", "look2")}
    kl1d = by["look2"]["all"] - base["by_arm"]["beam"]["all"]
    kl1b = by["look2"]["tok"] <= 2 * base["by_arm"]["beam"]["tok"]
    kl1 = ("PARTIAL" if 0 <= kl1d < 0.03 else
           "PASS-budget" if not kl1b else
           "PASS" if kl1d >= 0.03 else "FAIL")
    kl2d = by["look1"]["all"] - base["by_arm"]["walk"]["all"]
    onroute = float(np.mean([r["gold_on_route2"] for r in recs]))
    midonly = float(np.mean([
        r["gold_on_route2"] and r["look2"][0] == 0 for r in recs]))
    out = {"tag": "klookahead", "smoke": bool(limit),
           "limit": limit, "commit": CR.git_commit(),
           "model_fp": EC.fingerprint(), "ext_fp": efp,
           "baselines": {k: base["by_arm"][k]
                         for k in ("beam", "walk", "read", "flat")},
           "by_arm": by,
           "verdicts": {"KL1": [kl1, round(kl1d, 4), kl1b],
                        "KL2": [("PASS" if kl2d >= 0 else "FAIL"),
                                round(kl2d, 4)],
                        "KL3_descriptive":
                            {"gold_on_route2": round(onroute, 4),
                             "on_route_but_terminal_wrong":
                                 round(midonly, 4)}},
           "records": recs}
    name = "klook_smoke.json" if limit else "klookahead.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": by, "verdicts": out["verdicts"]},
                     indent=1))


# ------------------------------------------------------------------ score

def run_scored():
    g, adj = load_graph()
    with open(KQFILE) as f:
        qd = json.load(f)
    if "audit" not in qd:
        sys_msg = "question file lacks audit key — freeze first"
        raise SystemExit(sys_msg)
    qs = qd["eval"]
    cache, efp = embed_side(g, qs)
    log(f"embedded {g['n_entities']} entities + {g['n_triples']} triples "
        f"+ {len(qs)} questions (ext fp {efp})")

    recs = []
    for i, e in enumerate(qs):
        gold_n = norm_ent(e["gold"])
        fa, ft = arm_flat(e["q"], g)
        ra, rt = arm_read(e["q"], g, adj, cache)
        wa, wt, wh = arm_walk(e["q"], g, adj, cache, beam=False)
        ba, bt, bh = arm_walk(e["q"], g, adj, cache, beam=True)
        start_n = norm_ent(e["path"][0][0])
        recs.append({
            "q": e["q"], "gold": e["gold"], "hops": e["hops"],
            "recipe": e["recipe"], "entry_hit": int(any(
                norm_ent(x) == start_n for x in rank_entries(
                    cache, e["q"], [x for x in g["entities"]
                                    if adj[norm_ent(x)]])[0][:5])),
            "flat": [int(norm_ent(fa) == gold_n), fa, ft],
            "read": [int(norm_ent(ra) == gold_n), ra, rt],
            "walk": [int(norm_ent(wa) == gold_n), wa, wt, wh],
            "beam": [int(norm_ent(ba) == gold_n), ba, bt, bh]})
        log(f"  q{i + 1}/{len(qs)} h{e['hops']} {e['recipe']:8s} "
            f"flat={recs[-1]['flat'][0]} read={recs[-1]['read'][0]} "
            f"walk={recs[-1]['walk'][0]} beam={recs[-1]['beam'][0]}")

    def acc(arm, key=None, val=None):
        v = [r[arm][0] for r in recs
             if key is None or r[key] == val]
        return float(np.mean(v)) if v else None

    def toks(arm):
        return float(np.mean([r[arm][2] for r in recs]))

    by = {a: {"all": acc(a), "onehop": acc(a, "hops", 1),
              "twohop": acc(a, "hops", 2),
              "direct": acc(a, "recipe", "direct"),
              "indirect": acc(a, "recipe", "indirect"),
              "tok": toks(a)}
          for a in ("flat", "read", "walk", "beam")}
    k1d = by["walk"]["all"] - by["read"]["all"]
    k1_budget = by["walk"]["tok"] <= K1_GUARD * by["read"]["tok"]
    k1 = ("FAIL" if k1d < -0.02 else
          "FAIL-budget" if not k1_budget else
          "PASS" if k1d >= 0 else "PARTIAL")
    k2d = by["beam"]["all"] - by["walk"]["all"]
    k2_budget = by["beam"]["tok"] <= BEAM_GUARD * by["walk"]["tok"]
    k2 = ("FAIL" if k2d < 0.03 else
          "FAIL-budget" if not k2_budget else "PASS")
    bestb = max(by[a]["all"] for a in ("read", "walk", "beam"))
    k3d = by["flat"]["all"] - bestb
    k3 = ("PASS" if k3d >= 0 else
          "FAIL" if k3d < -0.02 else "PARTIAL")

    out = {"tag": "kgraphqa", "smoke": False, "commit": CR.git_commit(),
           "model_fp": EC.fingerprint(), "ext_fp": efp,
           "graph": {"entities": g["n_entities"],
                     "triples": g["n_triples"]},
           "by_arm": by,
           "entry_hit": float(np.mean([r["entry_hit"] for r in recs])),
           "verdicts": {"K1": [k1, round(k1d, 4), k1_budget],
                        "K2": [k2, round(k2d, 4), k2_budget],
                        "K3": [k3, round(k3d, 4)]},
           "records": recs}
    path = os.path.join(RES, "kgraphqa.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": by, "entry_hit": out["entry_hit"],
                      "verdicts": out["verdicts"]}, indent=1))


# ------------------------------------------------------------------ smoke

def run_smoke():
    cm = {"README.md": CR.build_chunkmap()["README.md"][:3]}
    raw = extract_triples(cm)
    case = defaultdict(Counter)
    for t in raw:
        for k in ("s", "o"):
            case[norm_ent(t[k])][t[k]] += 1
    canon = {k: c.most_common(1)[0][0] for k, c in case.items()}
    seen, tris = set(), []
    for t in raw:
        key = (norm_ent(t["s"]), t["r"].lower(), norm_ent(t["o"]))
        if key in seen:
            continue
        seen.add(key)
        tris.append({"s": canon[key[0]], "r": t["r"].lower(),
                     "o": canon[key[2]], "doc": t["doc"], "chunk": t["chunk"]})
    g = {"entities": sorted(canon.values()), "triples": tris,
         "n_entities": len(canon), "n_triples": len(tris)}
    adj = defaultdict(list)
    for ti, t in enumerate(tris):
        adj[norm_ent(t["s"])].append(ti)
        adj[norm_ent(t["o"])].append(ti)
    log(f"smoke graph: {g['n_entities']} entities, {g['n_triples']} triples")
    q = f"What is the lab rule about {tris[0]['o']}?" \
        if tris else "What is the lab rule?"
    fa, ft = arm_flat(q, g)
    out = {"tag": "kgraphqa", "smoke": True,
           "graph": [g["n_entities"], g["n_triples"]],
           "q": q, "flat_answer": fa, "flat_tok": ft}
    try:
        cache, fp = embed_side(g, [{"q": q}])
        ra, rt = arm_read(q, g, adj, cache)
        wa, wt, wh = arm_walk(q, g, adj, cache)
        out.update({"read_answer": ra, "read_tok": rt,
                    "walk_answer": wa, "walk_tok": wt, "walk_hops": wh,
                    "ext_fp": fp})
    except Exception as ex:  # noqa: BLE001
        out["embedder"] = f"unavailable: {ex}"
    with open(os.path.join(RES, "kgraphqa_smoke.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("smoke done (marked, never scored)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--look", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        run_smoke()
    elif a.build:
        build_graph(CR.build_chunkmap())
    elif a.gen:
        gen_questions()
    elif a.look:
        run_look(a.limit)
    else:
        run_scored()
