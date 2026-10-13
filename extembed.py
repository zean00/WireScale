"""E-series: a real RAG embedder (protocol.md, registered pre-data).

V-series exhausted the parasitic encoder's dials. E-series swaps the
retrieval arm to a DEDICATED embedding model — Qwen3-Embedding-0.6B
served in a remote container (via SSH tunnel on 127.0.0.1:8124) —
and re-asks the same frozen 58 on the pinned corpus. No judge calls,
no training. Separate cache file; the slot-7 cache is never touched.

    python extembed.py --smoke   # plumbing check, never scored
    python extembed.py           # scored run
"""
import argparse
import hashlib
import json
import os
import pickle
import time
import urllib.request

import numpy as np

import chunkrel as CR
import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
EXT_BASE = "http://127.0.0.1:8124"
EXT_MODEL = "qwen3-embedding-0.6b"
INSTRUCT = ("Instruct: Given a question, retrieve passages that "
            "answer it\nQuery: ")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ext_post(path, payload, timeout=120):
    req = urllib.request.Request(
        EXT_BASE + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ext_fingerprint():
    """Stable id of the external substrate: model id + live dim."""
    d = _ext_post("/v1/embeddings", {"input": ["fingerprint probe"],
                                      "model": EXT_MODEL})
    dim = len(d["data"][0]["embedding"])
    return (hashlib.sha1(f"{EXT_BASE}|{EXT_MODEL}|{dim}".encode())
            .hexdigest()[:12]), dim


def ext_cache_path(fp):
    return os.path.join(EC.CACHE_DIR, f"embext_{fp}.pkl")


def load_ext_cache(fp):
    try:
        with open(ext_cache_path(fp), "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def ext_embed(texts, cache, fp, batch=16):
    todo = [t for t in texts if t not in cache]
    log(f"external embed: {len(texts)} texts, {len(todo)} uncached")
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        d = _ext_post("/v1/embeddings",
                      {"input": part, "model": EXT_MODEL}, timeout=600)
        for t, item in zip(part, d["data"]):
            v = np.asarray(item["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:                      # server already unit-norms; enforce
                v /= n
            cache[t] = v
        if (i // batch) % 4 == 0:
            log(f"  {i + len(part)}/{len(todo)}")
    os.makedirs(EC.CACHE_DIR, exist_ok=True)
    with open(ext_cache_path(fp), "wb") as f:
        pickle.dump(cache, f)


def ext_vec(cache, t):
    return cache[t]


def _arm_metrics(questions, cm, qkey, cache, vecfn):
    """presence@1/@3 by recipe for one arm."""
    recs = []
    for e in questions:
        ch, gold = cm[e["doc"]], e["chunk"]
        qv = vecfn(cache, e[qkey])
        sims = np.asarray([float(vecfn(cache, t) @ qv) for t in ch])
        order = np.argsort(-sims)
        recs.append({"doc": e["doc"], "recipe": e["recipe"],
                     "p1": int(order[0] == gold),
                     "p3": int(gold in order[:3])})
    return recs


def agg(recs, key, recipe=None):
    vals = [r[key] for r in recs if recipe is None or r["recipe"] == recipe]
    return float(np.mean(vals)) if vals else None


def run_smoke():
    fp, dim = ext_fingerprint()
    cm_full = CR.build_chunkmap()
    cm = {"README.md": cm_full["README.md"][:5]}
    qs = [e for e in json.load(open(CR.QFILE))["eval"]
          if e["doc"] == "README.md" and e["chunk"] < 5][:2]
    cache = load_ext_cache(fp)
    texts = ([t for t in cm["README.md"]] + [e["q"] for e in qs]
             + [INSTRUCT + e["q"] for e in qs])
    ext_embed(texts, cache, fp)
    out = {"tag": "extembed", "smoke": True, "ext_fp": fp, "dim": dim,
           "qs": []}
    for e in qs:
        qv = cache[e["q"]]
        sims = [float(cache[t] @ qv) for t in cm["README.md"]]
        top = int(np.argmax(sims))
        qvi = cache[INSTRUCT + e["q"]]
        simsi = [float(cache[t] @ qvi) for t in cm["README.md"]]
        topi = int(np.argmax(simsi))
        out["qs"].append({"q": e["q"][:60], "gold": e["chunk"],
                          "e0_top": top, "e1_top": topi})
        log(f"  gold={e['chunk']} e0_top={top} e1_top={topi}  "
            f"{e['q'][:60]}")
    with open(os.path.join(RES, "extembed_smoke.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("smoke done (marked, never scored)")


def run_scored():
    with open(os.path.join(RES, "vautopsy.json")) as f:
        ref = json.load(f)["by_arm"]["V0"]["all"]["p3"]
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    cache7 = EC.load_cache()
    EC.embed(sorted({t for ch in cm.values() for t in ch}
                    | {e["q"] for e in qd["eval"]}), cache7)

    fp, dim = ext_fingerprint()
    xcache = load_ext_cache(fp)
    texts = sorted({t for ch in cm.values() for t in ch}
                   | {e["q"] for e in qd["eval"]}
                   | {INSTRUCT + e["q"] for e in qd["eval"]})
    ext_embed(texts, xcache, fp)

    evals = qd["eval"]
    arms = {
        "V0": _arm_metrics(evals, cm, "q", cache7, CR.vec),
        "E0": _arm_metrics(evals, cm, "q", xcache, ext_vec),
    }
    # E1 ranks with the instructed query but scores against plain chunks
    e1 = []
    for e in evals:
        ch, gold = cm[e["doc"]], e["chunk"]
        qv = xcache[INSTRUCT + e["q"]]
        sims = np.asarray([float(xcache[t] @ qv) for t in ch])
        order = np.argsort(-sims)
        e1.append({"doc": e["doc"], "recipe": e["recipe"],
                   "p1": int(order[0] == gold),
                   "p3": int(gold in order[:3])})
    arms["E1"] = e1

    by = {a: {rec: {"p1": agg(r, "p1", rec), "p3": agg(r, "p3", rec)}
              for rec in ("direct", "indirect")} for a, r in arms.items()}
    for a in by:
        by[a]["all"] = {"p1": agg(arms[a], "p1"), "p3": agg(arms[a], "p3")}

    sanity_ok = abs(by["V0"]["all"]["p3"] - ref) <= 0.03
    v0i = by["V0"]["indirect"]["p3"]
    best_e = max(("E0", "E1"), key=lambda a: by[a]["indirect"]["p3"])
    ea_gap = by[best_e]["indirect"]["p3"] - v0i
    ea = ("PASS" if ea_gap >= 0.10 else
          "PARTIAL" if ea_gap >= 0.03 else "FAIL")
    eb_d = by["E1"]["indirect"]["p3"] - by["E0"]["indirect"]["p3"]
    eb = ("instruct helps" if eb_d >= 0.03 else
          "instruct hurts" if eb_d <= -0.03 else "neutral")
    best_all = max(by[a]["all"]["p3"] for a in ("E0", "E1"))
    ec = "PASS" if best_all >= 2 * 0.128 else "FAIL"

    out = {"tag": "extembed", "smoke": False, "commit": CR.git_commit(),
           "model_fp": EC.fingerprint(),
           "ext_fp": fp, "ext_model": EXT_MODEL, "ext_dim": dim,
           "sanity": {"v0_p3": by["V0"]["all"]["p3"], "ref": ref,
                      "ok": sanity_ok},
           "by_arm": by, "records": arms,
           "verdicts": {"EA": ea, "ea_gap": round(ea_gap, 4),
                        "best_e_arm": best_e,
                        "EB": [eb, round(eb_d, 4)],
                        "EC": [ec, round(best_all, 4)]}}
    path = os.path.join(RES, "extembed.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": by, "verdicts": out["verdicts"],
                      "sanity": out["sanity"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run_smoke() if a.smoke else run_scored()
