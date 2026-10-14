"""Y-series: query rewriting for indirect questions (registered pre-data).

One LLM call (temp 0, seeded, slot 7) rewrites each frozen question
into documentation vocabulary; the E0 embedder ranks chunks against
the REWRITTEN query. Baselines read from committed extembed.json
(index-aligned per-question E0 records).

    python qrewrite.py --smoke   # 3 questions, never scored
    python qrewrite.py           # scored run (slot 7 + tunnel)
"""
import argparse
import json
import os
import time

import numpy as np

import chunkrel as CR

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RW_SYS = ("You rewrite a question into the vocabulary a technical "
          "documentation page would use. Output only the rewritten "
          "question. No preamble, no quotes.")
RW_USER = ("Question: {q}\n\nRewrite it using the concrete terms a "
           "documentation page would contain, so a search system can "
           "match it. Output only the rewritten question.")
RW_TEMP, RW_SEED, RW_MAXT = 0.0, 20261014, 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    with open(os.path.join(RES, "extembed.json")) as f:
        ex = json.load(f)
    import extembed as EE
    fp, _dim = EE.ext_fingerprint()
    xcache = EE.load_ext_cache(fp)
    evals = qd["eval"][:limit] if limit else qd["eval"]
    assert len(evals) <= len(ex["records"]["E0"]), \
        "E0 baseline misaligned with eval order"

    recs, i = [], 0
    rewrites = []
    for i, e in enumerate(evals):
        r = CR.llm_chat(
            [{"role": "system", "content": RW_SYS},
             {"role": "user", "content": RW_USER.format(q=e["q"])}],
            temp=RW_TEMP, max_tokens=RW_MAXT,
            seed=RW_SEED + 17 * i)
        q2 = r["choices"][0]["message"]["content"].strip().strip('"')
        rewrites.append(q2)
        log(f"  rewrite {i + 1}/{len(evals)}: {q2[:60]}")
    EE.ext_embed(sorted(set(rewrites)), xcache, fp)
    for i, (e, q2) in enumerate(zip(evals, rewrites)):
        ch = cm[e["doc"]]
        qv = xcache[q2]
        order = sorted(range(len(ch)),
                       key=lambda ci: -float(xcache[ch[ci]] @ qv))
        b0 = ex["records"]["E0"][i]
        recs.append({"doc": e["doc"], "recipe": e["recipe"],
                     "q": e["q"], "q2": q2,
                     "same": int(q2.strip().lower() == e["q"].strip().lower()),
                     "rw_p1": int(order[0] == e["chunk"]),
                     "rw_p3": int(e["chunk"] in order[:3]),
                     "e0_p1": b0["p1"], "e0_p3": b0["p3"]})
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"rw_p3={recs[-1]['rw_p3']} e0_p3={b0['p3']}  {q2[:50]}")

    def m(key, recipe=None):
        v = [r[key] for r in recs if recipe is None or r["recipe"] == recipe]
        return float(np.mean(v)) if v else None

    d_ind = m("rw_p3", "indirect") - m("e0_p3", "indirect")
    d_dir = m("rw_p3", "direct") - m("e0_p3", "direct")
    ya = ("PASS" if d_ind >= 0.05 else
          "PARTIAL" if d_ind >= 0 else "FAIL")
    yb = "PASS" if d_dir >= -0.03 else "FAIL"
    out = {"tag": "qrewrite", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": __import__(
               "embed_client").fingerprint(),
           "ext_fp": fp,
           "by": {"rw": {"direct": m("rw_p3", "direct"),
                         "indirect": m("rw_p3", "indirect"),
                         "all": m("rw_p3")},
                  "e0": {"direct": m("e0_p3", "direct"),
                         "indirect": m("e0_p3", "indirect"),
                         "all": m("e0_p3")}},
           "identical_frac": float(np.mean([r["same"] for r in recs])),
           "verdicts": {"YA": [ya, round(d_ind, 4)],
                        "YB": [yb, round(d_dir, 4)]},
           "records": recs}
    name = "qrewrite_smoke.json" if limit else "qrewrite.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by": out["by"], "identical_frac": out["identical_frac"],
                      "verdicts": out["verdicts"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(3 if a.smoke else 0)
