"""X-series: early-stop judging on embedder order (protocol.md, registered).

The W post-hoc sweep was order-free; this is the real sequential
procedure. Scan chunks in a registered order, judge each (CR.judge,
temp 0, slot 7), stop at the first p_yes >= .5. Flat baselines read
from committed walkret.json (its per-chunk p_yes doubles as the
cross-session sanity reference).

    python xstop.py --smoke   # 3 questions, never scored
    python xstop.py           # scored run (needs slot 7 + tunnel)
"""
import argparse
import json
import os
import time

import numpy as np

import chunkrel as CR
import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAU = 0.5
FLAT_RECALL, FLAT_TOK = 0.5172, 29236.0
XA_RECALL_RATIO, XA_TOK_RATIO = 0.95, 0.25
SANITY_MIN = 0.90


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ext_cache():
    import extembed as EE
    with open(os.path.join(RES, "extembed.json")) as f:
        fp = json.load(f)["ext_fp"]
    return EE.load_ext_cache(fp)


def run_scan(q, doc, order, cm, gold, ref_pyes):
    ch = cm[doc]
    judged_tok = 0
    agreements = []
    for n, ci in enumerate(order):
        yes, p, usage = CR.judge(q, ch[ci])
        judged_tok += usage.get("prompt_tokens", 0)
        ref = ref_pyes.get(str(ci))
        if ref is not None:
            agreements.append(int((p >= TAU) == (ref >= TAU)))
        stop = bool((p >= TAU) if p is not None else yes)
        if stop:
            return (int(ci == gold), int(gold in order[:n + 1]),
                    n + 1, judged_tok, agreements)
    return (0, int(gold in order), len(order), judged_tok, agreements)


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    with open(os.path.join(RES, "walkret.json")) as f:
        wr = json.load(f)
    xcache = load_ext_cache()
    evals = qd["eval"][:limit] if limit else qd["eval"]

    recs = []
    for i, e in enumerate(evals):
        ch = cm[e["doc"]]
        qv = xcache[e["q"]]
        emb_order = sorted(range(len(ch)),
                           key=lambda ci: -float(xcache[ch[ci]] @ qv))
        ref = next((r for r in wr["records"]
                    if r["doc"] == e["doc"] and r["q"] == e["q"]), None)
        ref_pyes = ref["p_yes"] if ref else {}
        s = run_scan(e["q"], e["doc"], emb_order, cm,
                     e["chunk"], ref_pyes)
        d = run_scan(e["q"], e["doc"], list(range(len(ch))), cm,
                     e["chunk"], ref_pyes)
        recs.append({
            "doc": e["doc"], "recipe": e["recipe"], "chunk": e["chunk"],
            "es": {"stop": s[0], "prefix": s[1], "n": s[2], "tok": s[3]},
            "ds": {"stop": d[0], "prefix": d[1], "n": d[2], "tok": d[3]},
            "agree": (float(np.mean(s[4])) if s[4] else None)})
        a = recs[-1]["agree"]
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"ES stop={s[0]} pref={s[1]} n={s[2]} | "
            f"DS stop={d[0]} pref={d[1]} n={d[2]} | agree="
            f"{a if a is None else round(a, 2)}")

    def m(arm, key):
        return float(np.mean([r[arm][key] for r in recs]))

    es_pref, es_stop, es_n, es_tok = (m("es", "prefix"), m("es", "stop"),
                                      m("es", "n"), m("es", "tok"))
    ds_stop, ds_n, ds_tok = m("ds", "stop"), m("ds", "n"), m("ds", "tok")
    agrees = [r["agree"] for r in recs if r["agree"] is not None]
    sanity = float(np.mean(agrees)) if agrees else None
    xa = ("PASS" if es_pref >= XA_RECALL_RATIO * FLAT_RECALL
          and es_tok <= XA_TOK_RATIO * FLAT_TOK else "FAIL")
    xb = "PASS" if es_stop >= ds_stop else "FAIL"
    gold_ranks = []
    for r, e in zip(recs, evals):
        ch = cm[e["doc"]]
        qv = xcache[e["q"]]
        order = sorted(range(len(ch)),
                       key=lambda ci: -float(xcache[ch[ci]] @ qv))
        gold_ranks.append(order.index(e["chunk"]) + 1)
    out = {"tag": "xstop", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "tau": TAU,
           "flat_committed": {"recall": FLAT_RECALL, "tok": FLAT_TOK},
           "es": {"prefix": es_pref, "stop": es_stop,
                  "n_judged": es_n, "tok": es_tok},
           "ds": {"prefix": m("ds", "prefix"), "stop": ds_stop,
                  "n_judged": ds_n, "tok": ds_tok},
           "sanity_agree": sanity, "sanity_ok":
               sanity is None or sanity >= SANITY_MIN,
           "gold_embedder_rank_mean": float(np.mean(gold_ranks)),
           "verdicts": {"XA": [xa, round(es_pref, 4),
                               round(es_tok / FLAT_TOK, 4)],
                        "XB": [xb, round(es_stop - ds_stop, 4)]},
           "records": recs}
    name = "xstop_smoke.json" if limit else "xstop.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("es", "ds", "sanity_agree", "sanity_ok",
                       "gold_embedder_rank_mean", "verdicts")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(3 if a.smoke else 0)
