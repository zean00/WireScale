"""V-series: the comparator autopsy (protocol.md, registered pre-data).

Why did bi-embed cosine rank at random within-doc in the R-series?
Pooling, input format, or length asymmetry — five arms on the SAME
frozen 58 eval questions. Mean-pooled arms read a second llama-server
(127.0.0.1:8999, --pooling mean, same weights, CPU-only); their
vectors live in a separate cache and never touch the shared slot-7
last-pool cache.

    python vautopsy.py --smoke   # plumbing check, never scored
    python vautopsy.py           # scored run (needs frozen questions)
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import urllib.request

import numpy as np

import chunkrel as CR
import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MEAN_BASE = "http://127.0.0.1:8999"
PFX_Q, PFX_C, LEAD = "query: ", "passage: ", 300


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_fingerprint():
    with urllib.request.urlopen(MEAN_BASE + "/props", timeout=10) as r:
        p = json.loads(r.read())
    return hashlib.sha1(p.get("model_path", "?").encode()).hexdigest()[:12]


def mean_cache_path():
    return os.path.join(EC.CACHE_DIR, f"embmean_{mean_fingerprint()}.pkl")


def load_mean_cache():
    try:
        with open(mean_cache_path(), "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def mean_embed(texts, cache):
    """Fill cache (text -> fp16 vec) from the 8999 mean-pool server."""
    todo = sorted({t for t in texts if t not in cache})
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        req = urllib.request.Request(
            MEAN_BASE + "/v1/embeddings",
            data=json.dumps({"input": batch, "model": "v-series",
                             "id_slot": 0}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        for item, t in zip(resp["data"], batch):
            cache[t] = np.asarray(item["embedding"], dtype=np.float16)
    if todo:
        os.makedirs(EC.CACHE_DIR, exist_ok=True)
        tmp = mean_cache_path() + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f, protocol=4)
        os.replace(tmp, mean_cache_path())
    return cache


def lead300(text):
    cut = text[:LEAD]
    snap = cut.rfind(" ", LEAD - 100, LEAD)
    return cut[:snap] if snap > 150 else cut


def run_smoke():
    cm = {"README.md": CR.build_chunkmap()["README.md"][:5]}
    cache = EC.load_cache()
    qd = json.load(open(CR.QFILE))
    evals = [e for e in qd["eval"] if e["doc"] == "README.md"][:2]
    EC.embed([e["q"] for e in evals] + [t for ch in cm.values() for t in ch]
             + [PFX_Q + e["q"] for e in evals]
             + [PFX_C + t for ch in cm.values() for t in ch], cache)
    out = {"tag": "vautopsy", "smoke": True, "questions": []}
    for e in evals:
        qv0 = CR.vec(cache, e["q"])
        qv1 = CR.vec(cache, PFX_Q + e["q"])
        for name, qv, tf in (("V0", qv0, lambda t: t),
                             ("V1", qv1, lambda t: PFX_C + t)):
            mat = np.stack([CR.vec(cache, tf(t)) for t in cm["README.md"]])
            order = np.argsort(-(mat @ qv))
            log(f"  {name} gold={e['chunk']} top1={int(order[0])} "
                f"p3={int(e['chunk'] in order[:3])}")
            out["questions"].append({"arm": name, "gold": e["chunk"],
                                     "top1": int(order[0]),
                                     "p3": int(e["chunk"] in order[:3])})
    try:
        mc = mean_embed(["smoke probe"], load_mean_cache())
        v = mc["smoke probe"].astype(np.float32)
        log(f"  mean server OK: dim={v.shape[0]} "
            f"L2={float(np.linalg.norm(v)):.3f}")
        out["mean_server"] = "ok"
    except Exception as e:  # noqa: BLE001
        log(f"  mean server probe failed (OK in smoke): {e}")
        out["mean_server"] = f"unavailable: {e}"
    with open(os.path.join(RES, "vautopsy_smoke.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("smoke done (marked, never scored)")


def run_scored():
    if not os.path.exists(CR.QFILE):
        sys.exit("frozen question file missing — generate/audit first")
    with open(os.path.join(RES, "chunkrel.json")) as f:
        ref_v3 = json.load(f)["by_recipe"]["all"]["v_p3"]
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    evals = qd["eval"]
    qtexts = [e["q"] for e in evals]
    chunks = [t for ch in cm.values() for t in ch]

    cache = EC.load_cache()
    ec_texts = sorted(set(qtexts + chunks
                          + [PFX_Q + q for q in qtexts]
                          + [PFX_C + c for c in chunks]
                          + [lead300(c) for c in chunks]))
    log(f"embedding {len(ec_texts)} last-pool texts (slot 7)")
    EC.embed(ec_texts, cache)
    mcache = mean_embed(sorted(set(qtexts + chunks
                                   + [PFX_Q + q for q in qtexts]
                                   + [PFX_C + c for c in chunks])),
                        load_mean_cache())
    log(f"mean cache: {len(mcache)} vectors")

    def v_ec(t):
        v = cache[t].astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def v_mean(t):
        v = mcache[t].astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-8)

    arms = {
        "V0": (lambda e: e["q"], lambda t: t, v_ec),
        "V1": (lambda e: PFX_Q + e["q"], lambda t: PFX_C + t, v_ec),
        "V2": (lambda e: e["q"], lambda t: t, v_mean),
        "V3": (lambda e: PFX_Q + e["q"], lambda t: PFX_C + t, v_mean),
        "V4": (lambda e: e["q"], lead300, v_ec),
    }
    mats = {name: {d: np.stack([vf(cf(t)) for t in ch])
                   for d, ch in cm.items()}
            for name, (_, cf, vf) in arms.items()}

    records = []
    for i, e in enumerate(evals):
        rec = {"doc": e["doc"], "chunk": e["chunk"],
               "recipe": e["recipe"], "q": e["q"]}
        for name, (qf, _, vf) in arms.items():
            order = np.argsort(-(mats[name][e["doc"]] @ vf(qf(e))))
            rec[f"{name}_p1"] = int(order[0] == e["chunk"])
            rec[f"{name}_p3"] = int(e["chunk"] in order[:3])
        records.append(rec)
        log(f"  q{i + 1}/{len(evals)} " +
            " ".join(f"{n}={rec[f'{n}_p3']}" for n in arms))

    def agg(key, recipe=None):
        vals = [r[key] for r in records
                if recipe is None or r["recipe"] == recipe]
        return float(np.mean(vals)) if vals else None

    by = {}
    for name in arms:
        by[name] = {"direct": {"p1": agg(f"{name}_p1", "direct"),
                               "p3": agg(f"{name}_p3", "direct")},
                    "indirect": {"p1": agg(f"{name}_p1", "indirect"),
                                 "p3": agg(f"{name}_p3", "indirect")},
                    "all": {"p1": agg(f"{name}_p1"),
                            "p3": agg(f"{name}_p3")}}
    sanity_ok = abs(by["V0"]["all"]["p3"] - ref_v3) <= 0.03

    gaps = {n: by[n]["indirect"]["p3"] - by["V0"]["indirect"]["p3"]
            for n in arms if n != "V0"}
    best = max(gaps, key=gaps.get)
    va = ("PASS" if gaps[best] >= 0.10 else
          "PARTIAL" if gaps[best] >= 0.03 else "FAIL")
    mean_eff = ((by["V2"]["indirect"]["p3"] + by["V3"]["indirect"]["p3"])
                / 2
                - (by["V0"]["indirect"]["p3"]
                   + by["V1"]["indirect"]["p3"]) / 2)
    pfx_eff = ((by["V1"]["indirect"]["p3"] + by["V3"]["indirect"]["p3"])
               / 2
               - (by["V0"]["indirect"]["p3"]
                  + by["V2"]["indirect"]["p3"]) / 2)
    driver = ("pooling" if mean_eff > pfx_eff else
              "format" if pfx_eff > mean_eff else "neither")
    if max(mean_eff, pfx_eff) < 0.03:
        driver = "neither"
    vc_gap = by["V4"]["direct"]["p3"] - by["V0"]["direct"]["p3"]

    out = {"tag": "vautopsy", "smoke": False, "commit": CR.git_commit(),
           "model_fp": EC.fingerprint(), "mean_fp": mean_fingerprint(),
           "sanity": {"v0_p3": by["V0"]["all"]["p3"], "ref": ref_v3,
                      "ok": sanity_ok},
           "by_arm": by, "records": records,
           "verdicts": {"VA": va, "best_arm": best,
                        "best_gap": round(gaps[best], 4),
                        "VB": {"mean_effect": round(mean_eff, 4),
                               "prefix_effect": round(pfx_eff, 4),
                               "driver": driver},
                        "VC": ("PASS" if vc_gap >= 0.05 else "FAIL",
                               round(vc_gap, 4))}}
    path = os.path.join(RES, "vautopsy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": {n: by[n] for n in arms},
                      "verdicts": out["verdicts"],
                      "sanity": out["sanity"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run_smoke() if a.smoke else run_scored()
