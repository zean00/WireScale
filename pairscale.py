"""J-series: pair-head scaling (protocol.md, registered pre-data).

R2's FAIL came at 230 train pairs. Sample size or mechanism? Generate
8 questions per (train chunk, recipe) with the R-series generator
machinery (temp .8, NEW seed base, slot 7), retrain J-combo unchanged,
evaluate on the SAME frozen 58 (never trained on). Per-chunk scores
are persisted per question (R's runner discarded them). Train output
is UNAUDITED generator output — registered disclosure.

    python pairscale.py --smoke   # plumbing check, never scored
    python pairscale.py --gen     # write results/j_train8.json
    python pairscale.py           # scored run (needs --gen output)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import chunkrel as CR
import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
JFILE = os.path.join(RES, "j_train8.json")
J_GEN_SEED, N_PER = 20261013, 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_train8():
    cm = CR.build_chunkmap()
    out, i = [], 0
    for doc in CR.DOCS:
        if doc not in cm:
            continue
        tr, _ = CR.split_indices(len(cm[doc]))
        for ci in tr:
            for recipe, tmpl in CR.RECIPES.items():
                for _k in range(N_PER):
                    r = CR.llm_chat(
                        [{"role": "system", "content": CR.GEN_SYS},
                         {"role": "user", "content":
                          tmpl.format(chunk=cm[doc][ci])}],
                        temp=CR.GEN_TEMP, max_tokens=64,
                        seed=J_GEN_SEED * 100003 + 7 * i)
                    i += 1
                    q = r["choices"][0]["message"]["content"].strip().strip('"')
                    out.append({"doc": doc, "chunk": ci,
                                "recipe": recipe, "q": q})
                    if i % 40 == 0:
                        log(f"  {i} generated")
    data = {"train": out, "n_per": N_PER, "commit": CR.git_commit(),
            "model_fp": EC.fingerprint(),
            "note": ("J-series train-only generator output, UNAUDITED "
                     "(registered disclosure; R-series train policy)")}
    with open(JFILE, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    log(f"wrote {JFILE} — {len(out)} train questions (unaudited, "
        "train-only)")


def _eval_net(net, cm, cache, e):
    ch, gold = cm[e["doc"]], e["chunk"]
    qv = CR.vec(cache, e["q"])
    F = np.stack([CR.feats(qv, CR.vec(cache, t)) for t in ch])
    with torch.no_grad():
        s = torch.sigmoid(net(torch.from_numpy(F))).numpy()
    o = np.argsort(-s)
    return (int(o[0] == gold), int(gold in o[:3]),
            [round(float(x), 4) for x in s])


def run_smoke():
    qd = json.load(open(CR.QFILE))
    cm = {"README.md": CR.build_chunkmap()["README.md"][:5]}
    qd = {"train": [e for e in qd["train"]
                    if e["doc"] == "README.md" and e["chunk"] < 5],
          "eval": [e for e in qd["eval"]
                   if e["doc"] == "README.md" and e["chunk"] < 5]}
    cache = EC.load_cache()
    evals = qd["eval"][:2]
    p1 = CR.build_pairs(qd, cm, cache)
    net, ce = CR.train_combo(p1, cache)
    out = {"tag": "pairscale", "smoke": True, "val_ce": ce, "qs": []}
    for e in evals:
        a, b, _ = _eval_net(net, cm, cache, e)
        out["qs"].append({"q": e["q"][:60], "p1": a, "p3": b})
        log(f"  p1={a} p3={b}  {e['q'][:60]}")
    with open(os.path.join(RES, "pairscale_smoke.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log("smoke done (marked, never scored)")


def run_scored():
    if not os.path.exists(CR.QFILE):
        sys.exit("frozen question file missing")
    if not os.path.exists(JFILE):
        sys.exit("j_train8.json missing — run --gen first")
    with open(os.path.join(RES, "chunkrel.json")) as f:
        ref = json.load(f)["by_recipe"]
    qd = json.load(open(CR.QFILE))
    j8d = json.load(open(JFILE))
    cm = CR.build_chunkmap()
    cache = EC.load_cache()
    texts = sorted({t for ch in cm.values() for t in ch}
                   | {e["q"] for e in qd["train"]}
                   | {e["q"] for e in j8d["train"]}
                   | {e["q"] for e in qd["eval"]})
    EC.embed(texts, cache)

    p1 = CR.build_pairs(qd, cm, cache)
    p8 = CR.build_pairs(j8d, cm, cache)
    log(f"pairs: J1={len(p1)} J8={len(p8)}")
    net1, ce1 = CR.train_combo(p1, cache)
    net8, ce8 = CR.train_combo(p8, cache)
    log(f"J1 valBCE={ce1:.4f}  J8 valBCE={ce8:.4f}")

    records = []
    for i, e in enumerate(qd["eval"]):
        ch, gold = cm[e["doc"]], e["chunk"]
        qv = CR.vec(cache, e["q"])
        mat = np.stack([CR.vec(cache, t) for t in ch])
        order = np.argsort(-(mat @ qv))
        j1p1, j1p3, _ = _eval_net(net1, cm, cache, e)
        j8p1, j8p3, scores = _eval_net(net8, cm, cache, e)
        records.append({"doc": e["doc"], "chunk": gold,
                        "recipe": e["recipe"], "q": e["q"],
                        "v_p1": int(order[0] == gold),
                        "v_p3": int(gold in order[:3]),
                        "j1_p1": j1p1, "j1_p3": j1p3,
                        "j8_p1": j8p1, "j8_p3": j8p3,
                        "j8_scores": scores})
        log(f"  q{i + 1}/58 v3={records[-1]['v_p3']} "
            f"j1={j1p3} j8={j8p3}")

    def agg(key, recipe=None):
        vals = [r[key] for r in records
                if recipe is None or r["recipe"] == recipe]
        return float(np.mean(vals)) if vals else None

    by = {arm: {rec: {"p1": agg(f"{arm}_p1", rec), "p3": agg(f"{arm}_p3", rec)}
                for rec in ("direct", "indirect")}
          for arm in ("v", "j1", "j8")}
    for arm in ("v", "j1", "j8"):
        by[arm]["all"] = {"p1": agg(f"{arm}_p1"), "p3": agg(f"{arm}_p3")}

    sanity_ok = abs(by["j1"]["all"]["p3"] - ref["all"]["c_p3"]) <= 0.03
    ja_gap = by["j8"]["all"]["p3"] - by["v"]["all"]["p3"]
    ja = "PASS" if ja_gap >= 0.05 else "PARTIAL" if ja_gap >= 0 else "FAIL"
    jb = ("PASS" if by["j8"]["indirect"]["p3"]
          >= 0.5 * by["j8"]["direct"]["p3"] else "FAIL")

    out = {"tag": "pairscale", "smoke": False, "commit": CR.git_commit(),
           "model_fp": EC.fingerprint(),
           "n_train": {"j1": len(qd["train"]), "j8": len(j8d["train"])},
           "n_pairs": {"j1": len(p1), "j8": len(p8)},
           "val_ce": {"j1": ce1, "j8": ce8},
           "sanity": {"j1_p3": by["j1"]["all"]["p3"],
                      "ref_c_p3": ref["all"]["c_p3"], "ok": sanity_ok},
           "by_arm": by, "records": records,
           "verdicts": {"JA": ja, "ja_gap": round(ja_gap, 4), "JB": jb}}
    path = os.path.join(RES, "pairscale.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": by, "verdicts": out["verdicts"],
                      "sanity": out["sanity"],
                      "n_pairs": out["n_pairs"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gen", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run_smoke()
    elif a.gen:
        gen_train8()
    else:
        run_scored()
