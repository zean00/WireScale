"""C-series: the pool-confined machine — RAG retrieves, the machine
reads (protocol.md, registered pre-data).

B-series showed the top-8 pool holds the gold page .810 of the time
yet pile synthesis flatlines, while the machine's verified
single-page reader synthesizes .852 at 5.4k tok/q. The disclosed
post-hoc simulation (registration) says confining the machine's
scan-and-stop to the committed top-8 pool costs zero chunk-level
accuracy at ~3.6 judged pages/question. This scores that live:
ASM-8 (primary, bars C1-C3) and ASM-4 (secondary, descriptive),
judgments shared, answers separate, sim-vs-live agreement recorded.

    python poolstop.py --smoke   # 2 questions, never scored
    python poolstop.py           # scored run (needs slot 7; no tunnel)
"""
import argparse
import json
import os
import time

import numpy as np

import asm
import chunkrel as CR
import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAU = 0.5
C1_BAND = 0.02
C1_PARTIAL = 0.04
C2_RATIO = 0.5
C3_PARTIAL = 0.02
SANITY_MIN = 0.95


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_pool(q, ch, pool, ref_pyes):
    """Live judge in pool order; stop at first cal p >= TAU.
    Returns (stop_ci or None, judged, agreements)."""
    judged, agreements = [], []
    for ci in pool:
        yes, p, usage = CR.judge(q, ch[ci])
        judged.append({"ci": ci, "p": (round(float(p), 4)
                                       if p is not None else None),
                       "yes": bool(yes),
                       "tok": usage.get("prompt_tokens", 0)})
        d = asm.cal_p(p)
        stop = bool((d >= TAU) if d is not None else yes)
        ref = ref_pyes.get(str(ci))
        if ref is not None:
            agreements.append(int(stop == (ref >= TAU)))
        if stop:
            return ci, judged, agreements
    return None, judged, agreements


def arm4_decision(judged, pool):
    """ASM-4 from the shared judged prefix (first 4 pool pages)."""
    for j in judged:
        if j["ci"] not in pool[:4]:
            break
        d = asm.cal_p(j["p"]) if j["p"] is not None else None
        stop = bool((d >= TAU) if d is not None else j["yes"])
        if stop:
            return j["ci"]
    return None


def sim_stop(pool, ps):
    for c in pool:
        p = ps.get(str(c))
        if p is not None and asm.cal_p(p) >= TAU:
            return c
    return None


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    with open(os.path.join(RES, "topk.json")) as f:
        topk = json.load(f)
    with open(os.path.join(RES, "assembly.json")) as f:
        committed = json.load(f)
    with open(os.path.join(RES, "walkret.json")) as f:
        wr = json.load(f)
    pools = {(r["doc"], r["q"]): r["topk8"]["pick"]
             for r in topk["records"]}
    pmap = {(r["doc"], r["q"]): r["p_yes"] for r in wr["records"]}
    ASM_ACC = committed["by_arm"]["asm"]["ans_acc"]
    ASM_TOK = committed["by_arm"]["asm"]["tok_per_q"]
    TOPK2_ACC = topk["by_k"]["2"]["ans_acc"]

    evals = qd["eval"][:limit] if limit else qd["eval"]
    recs = []
    for i, e in enumerate(evals):
        ch = cm[e["doc"]]
        gold, q = e["chunk"], e["q"]
        pool8 = pools[(e["doc"], q)]
        ps = pmap.get((e["doc"], q), {})
        seed = asm.GEN_SEED_BASE + 17 * i

        stop8, judged, agrs = scan_pool(q, ch, pool8, ps)
        fb8 = stop8 is None
        ci8 = stop8 if stop8 is not None else pool8[0]
        a8 = asm.answer_arm(q, ch, ci8, gold, seed)
        stop4 = arm4_decision(judged, pool8)
        ci4 = stop4 if stop4 is not None else pool8[0]
        a4 = (a8 if ci4 == ci8
              else asm.answer_arm(q, ch, ci4, gold, seed))
        tok4 = sum(j["tok"] for j in judged[:4])

        sim8, sim4 = sim_stop(pool8, ps), sim_stop(pool8[:4], ps)
        sim8_ci = sim8 if sim8 is not None else pool8[0]
        sim4_ci = sim4 if sim4 is not None else pool8[0]
        recs.append({
            "doc": e["doc"], "recipe": e["recipe"], "chunk": gold, "q": q,
            "pool": pool8,
            "asm8": {**a8, "fallback": int(fb8), "stop_ci": stop8,
                     "n_judged": len(judged),
                     "judged": judged,
                     "judge_tok": sum(j["tok"] for j in judged),
                     "total_tok": sum(j["tok"] for j in judged) + a8["tok"],
                     "sim_ci": sim8_ci,
                     "sim_agree": int(ci8 == sim8_ci)},
            "asm4": {**a4, "fallback": int(stop4 is None),
                     "stop_ci": stop4,
                     "n_judged": min(4, len(judged))
                     if stop4 is None else
                     next(j + 1 for j, jj in enumerate(judged)
                          if jj["ci"] == stop4),
                     "total_tok": tok4 + a4["tok"],
                     "sim_ci": sim4_ci,
                     "sim_agree": int(ci4 == sim4_ci)},
            "chunk_agree": (float(np.mean(agrs)) if agrs else None)})
        r8 = recs[-1]["asm8"]
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"ASM8 hit={r8['hit']} ok={r8['ans_ok']} "
            f"fb={r8['fallback']} n={r8['n_judged']} | "
            f"ASM4 hit={recs[-1]['asm4']['hit']} "
            f"ok={recs[-1]['asm4']['ans_ok']} | "
            f"sim_agree={r8['sim_agree']}")

    def m(arm, key, subset=None):
        rs = [r[arm][key] for r in recs
              if subset is None or r["recipe"] == subset]
        return float(np.mean(rs)) if rs else None

    by_arm = {}
    for arm in ("asm8", "asm4"):
        by_arm[arm] = {
            "ans_acc": m(arm, "ans_ok"),
            "stop_hit": m(arm, "hit"),
            "fallback_rate": m(arm, "fallback"),
            "n_judged": m(arm, "n_judged"),
            "tok_per_q": m(arm, "total_tok"),
            "grade_p": m(arm, "grade_p"),
            "per_recipe": {rc: m(arm, "ans_ok", rc)
                           for rc in ("direct", "indirect")}}
    # fallback vs stopped answer quality (A-series descriptive pattern)
    fb_ok = [r["asm8"]["ans_ok"] for r in recs if r["asm8"]["fallback"]]
    st_ok = [r["asm8"]["ans_ok"] for r in recs
             if not r["asm8"]["fallback"]]
    fb_gp = [r["asm8"]["grade_p"] for r in recs
             if r["asm8"]["fallback"] and r["asm8"]["grade_p"] is not None]
    st_gp = [r["asm8"]["grade_p"] for r in recs
             if not r["asm8"]["fallback"]
             and r["asm8"]["grade_p"] is not None]

    ags = [r["chunk_agree"] for r in recs if r["chunk_agree"] is not None]
    chunk_agree = float(np.mean(ags)) if ags else None
    sim_agree8 = m("asm8", "sim_agree")
    sim_agree4 = m("asm4", "sim_agree")

    a8_acc = by_arm["asm8"]["ans_acc"]
    a8_tok = by_arm["asm8"]["tok_per_q"]
    d1 = a8_acc - ASM_ACC
    c1 = ("PASS" if d1 >= -C1_BAND else
          "PARTIAL" if d1 >= -C1_PARTIAL else "FAIL")
    c2 = ("PASS-budget" if a8_tok <= C2_RATIO * ASM_TOK
          else "FAIL-budget")
    d3 = a8_acc - TOPK2_ACC
    c3 = ("PASS" if d3 >= 0 else
          "PARTIAL" if d3 >= -C3_PARTIAL else "FAIL")
    out = {"tag": "poolstop", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "tau": TAU,
           "committed_anchors": {
               "asm_acc": ASM_ACC, "asm_tok": ASM_TOK,
               "topk2_acc": TOPK2_ACC,
               "c2_ceiling_tok": C2_RATIO * ASM_TOK},
           "registered_sim_predictions": {
               "stop_hit_27_of_58_band": [26, 28],
               "fallback_16_of_58_band": [14, 18],
               "mean_judged_3p6_band": [3.4, 3.9]},
           "by_arm": by_arm,
           "asm8_descriptives": {
               "fb_ans_ok": (float(np.mean(fb_ok)) if fb_ok else None),
               "stopped_ans_ok": (float(np.mean(st_ok)) if st_ok else None),
               "fb_grade_p": (float(np.mean(fb_gp)) if fb_gp else None),
               "stopped_grade_p": (float(np.mean(st_gp))
                                   if st_gp else None),
               "fb_n": len(fb_ok)},
           "sanity": {"chunk_agree": chunk_agree,
                      "sim_answered_agree_8": sim_agree8,
                      "sim_answered_agree_4": sim_agree4,
                      "ok": chunk_agree is None
                      or chunk_agree >= SANITY_MIN},
           "verdicts": {"C1": [c1, round(d1, 4)],
                        "C2": [c2, round(a8_tok, 1)],
                        "C3": [c3, round(d3, 4)]},
           "records": recs}
    name = "poolstop_smoke.json" if limit else "poolstop.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("by_arm", "asm8_descriptives", "sanity",
                       "verdicts")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
