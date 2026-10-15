"""D-series: strengthening the verifier (protocol.md, registered pre-data).

The C-series named the binding constraint: the top-8 pool holds gold
.810 of the time, the verifier captures .466. Disclosed diagnostic:
the judge misses gold CONFIDENTLY (9/11 judged misses below .20 cal)
— sampling variance is not expected to rescue it. This scores two
verifier instruments on the full pool: S = single temp-0 call
(committed instrument), M = 3-sample temp-1.0 mean; arms D-M
(primary), D-H (high bar), D-50 (replication sanity) + a chunk-level
tau sweep.

    python pooljudge.py --smoke   # 2 questions, never scored
    python pooljudge.py           # scored run (needs slot 7; no tunnel)
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
HIGH_TAU = 0.70
SWEEP_TAU = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
N_SAMPLES = 3
SAMP_SEED_BASE = 20261015
ASM8_TOK_COMMITTED = 5427.672413793103
SANITY_MIN = 0.95

JUDGE_USER = ("Question: {q}\n\nPassage:\n{c}\n\n"
              "Does the passage contain information that helps answer "
              "the question?")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def judge_t(q, chunk, temp, seed=None):
    """CR.judge with explicit temp/seed; prompt byte-identical."""
    r = CR.llm_chat([{"role": "system", "content": CR.JUDGE_SYS},
                     {"role": "user", "content":
                      JUDGE_USER.format(q=q, c=chunk)}],
                    temp=temp, max_tokens=CR.LLM_MAXT, logprobs=True,
                    seed=seed)
    p = CR.p_yes_of(r)
    content = r["choices"][0]["message"]["content"].strip().lower()
    yes = (p >= 0.5) if p is not None else \
        bool(content) and content.split()[0].startswith("y")
    return yes, p, r.get("usage", {})


def stop_of(p, tau):
    d = asm.cal_p(p)
    return bool((d >= tau) if d is not None else False)


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    with open(os.path.join(RES, "topk.json")) as f:
        topk = json.load(f)
    with open(os.path.join(RES, "poolstop.json")) as f:
        committed = json.load(f)
    with open(os.path.join(RES, "walkret.json")) as f:
        wr = json.load(f)
    pools = {(r["doc"], r["q"]): r["topk8"]["pick"]
             for r in topk["records"]}
    committed_ci = {(r["doc"], r["q"]): r["asm8"]["ci"]
                    for r in committed["records"]}
    ASM8_ACC = committed["by_arm"]["asm8"]["ans_acc"]
    ASM8_HIT = committed["by_arm"]["asm8"]["stop_hit"]
    pmap = {(r["doc"], r["q"]): r["p_yes"] for r in wr["records"]}
    evals = qd["eval"][:limit] if limit else qd["eval"]

    recs = []
    for i, e in enumerate(evals):
        ch = cm[e["doc"]]
        gold, q = e["chunk"], e["q"]
        pool = pools[(e["doc"], q)]
        ps = pmap.get((e["doc"], q), {})
        seed = asm.GEN_SEED_BASE + 17 * i

        s_vec, m_vec, agrs = [], [], []
        for ci in pool:
            y, p, u = judge_t(q, ch[ci], 0)
            ref = ps.get(str(ci))
            if ref is not None:
                d = asm.cal_p(p)
                stop = bool((d >= TAU) if d is not None else y)
                agrs.append(int(stop == (ref >= TAU)))
            s_vec.append({"ci": ci, "p": p, "tok": u.get("prompt_tokens", 0)})
            ps3 = []
            for s in range(N_SAMPLES):
                _, p3, u3 = judge_t(q, ch[ci], 1.0,
                                    SAMP_SEED_BASE + 1013 * i + 17 * s)
                ps3.append(p3)
            ps3 = [x for x in ps3 if x is not None]
            m_vec.append({"ci": ci, "p": (float(np.mean(ps3))
                                          if ps3 else None),
                          "p3": [round(float(x), 4) for x in ps3],
                          "tok": u.get("prompt_tokens", 0)})

        # page prompt tokens are identical across samples and temps, so
        # deployed M cost = per-page tok x N_SAMPLES (see deployed_tok)

        def scan(vec, tau):
            for v in vec:
                if v["p"] is not None and stop_of(v["p"], tau):
                    return v["ci"]
            return None

        d50 = scan(s_vec, TAU)
        dh = scan(s_vec, HIGH_TAU)
        dm = scan(m_vec, TAU)
        ci50 = d50 if d50 is not None else pool[0]
        ci_h = dh if dh is not None else pool[0]
        ci_m = dm if dm is not None else pool[0]
        a_m = asm.answer_arm(q, ch, ci_m, gold, seed)
        a_h = asm.answer_arm(q, ch, ci_h, gold, seed)
        a50 = None
        if ci50 != committed_ci[(e["doc"], q)]:
            a50 = asm.answer_arm(q, ch, ci50, gold, seed)

        def deployed_tok(vec, ci, samples):
            toks = [v["tok"] for v in vec]
            pos = next((j for j, v in enumerate(vec) if v["ci"] == ci),
                       len(vec))
            return sum(toks[:pos + 1]) * samples

        recs.append({
            "doc": e["doc"], "recipe": e["recipe"], "chunk": gold, "q": q,
            "pool": pool,
            "s_vec": s_vec, "m_vec": m_vec,
            "d50": {"ci": ci50, "stop_ci": d50,
                    "fallback": int(d50 is None),
                    "hit": int(ci50 == gold),
                    "matches_committed": int(ci50 == committed_ci[(e["doc"], q)]),
                    "answer": a50},
            "dh": {**a_h, "stop_ci": dh, "fallback": int(dh is None),
                   "n_judged": (next((j + 1 for j, v in enumerate(s_vec)
                                      if v["ci"] == ci_h), len(s_vec))),
                   "deployed_judge_tok": deployed_tok(s_vec, ci_h, 1),
                   "total_tok": deployed_tok(s_vec, ci_h, 1) + a_h["tok"]},
            "dm": {**a_m, "stop_ci": dm, "fallback": int(dm is None),
                   "n_judged": (next((j + 1 for j, v in enumerate(m_vec)
                                      if v["ci"] == ci_m), len(m_vec))),
                   "deployed_judge_tok": deployed_tok(m_vec, ci_m, N_SAMPLES),
                   "total_tok": deployed_tok(m_vec, ci_m, N_SAMPLES)
                   + a_m["tok"]}})
        r = recs[-1]
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"DM hit={r['dm']['hit']} ok={r['dm']['ans_ok']} "
            f"fb={r['dm']['fallback']} | "
            f"DH hit={r['dh']['hit']} ok={r['dh']['ans_ok']} "
            f"fb={r['dh']['fallback']} | "
            f"d50_match={r['d50']['matches_committed']} "
            f"agree={round(float(np.mean(agrs)), 3) if agrs else None}")

    def m(arm, key, subset=None):
        rs = [r[arm][key] for r in recs
              if subset is None or r["recipe"] == subset]
        return float(np.mean(rs)) if rs else None

    # chunk-level tau sweep (descriptive): first-cross capture + fallback hit
    sweep = []
    for tau in SWEEP_TAU:
        for inst, vec_key in (("S", "s_vec"), ("M", "m_vec")):
            cap = fb = fb_hit = nj = 0
            for r in recs:
                stop = None
                for j, v in enumerate(r[vec_key]):
                    if v["p"] is not None and stop_of(v["p"], tau):
                        stop = v["ci"]
                        nj += j + 1
                        break
                if stop is None:
                    fb += 1
                    fb_hit += int(r["pool"][0] == r["chunk"])
                    nj += len(r[vec_key])
                else:
                    cap += int(stop == r["chunk"])
            n = len(recs)
            sweep.append({"tau": tau, "inst": inst,
                          "capture": round(cap / n, 4),
                          "fallback_rate": round(fb / n, 4),
                          "fb_hit": round(fb_hit / n, 4),
                          "mean_n_judged": round(nj / n, 2)})

    d50_match = m("d50", "matches_committed")
    ags = []
    for r in recs:
        ps = pmap.get((r["doc"], r["q"]), {})
        for v in r["s_vec"]:
            ref = ps.get(str(v["ci"]))
            if ref is not None and v["p"] is not None:
                ags.append(int(stop_of(v["p"], TAU) == (ref >= TAU)))
    chunk_agree = float(np.mean(ags)) if ags else None

    dm_acc, dm_hit = m("dm", "ans_ok"), m("dm", "hit")
    dh_acc = m("dh", "ans_ok")
    dm_tok = m("dm", "total_tok")
    d1 = ("PASS" if dm_hit >= ASM8_HIT + 0.05 else
          "PARTIAL" if dm_hit >= ASM8_HIT + 0.02 else "FAIL")
    d2 = ("PASS" if dm_acc >= ASM8_ACC + 0.03 else
          "PARTIAL" if dm_acc >= ASM8_ACC - 0.02 else "FAIL")
    d3 = ("PASS-budget" if dm_tok <= ASM8_TOK_COMMITTED
          else "FAIL-budget")
    d4 = ("PASS" if dh_acc >= ASM8_ACC else
          "PARTIAL" if dh_acc >= ASM8_ACC - 0.02 else "FAIL")
    out = {"tag": "pooljudge", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "n_samples": N_SAMPLES, "sweep_tau": list(SWEEP_TAU),
           "committed_anchors": {"asm8_acc": ASM8_ACC,
                                 "asm8_hit": ASM8_HIT,
                                 "asm8_tok": ASM8_TOK_COMMITTED},
           "by_arm": {a: {"ans_acc": m(a, "ans_ok"),
                          "stop_hit": m(a, "hit"),
                          "fallback_rate": m(a, "fallback"),
                          "n_judged": m(a, "n_judged"),
                          "tok_per_q": m(a, "total_tok"),
                          "grade_p": m(a, "grade_p"),
                          "per_recipe": {rc: m(a, "ans_ok", rc)
                                         for rc in ("direct", "indirect")}}
                      for a in ("dm", "dh")},
           "d50_agree_committed": d50_match,
           "sanity": {"chunk_agree_S": chunk_agree,
                      "d50_agree": d50_match,
                      "ok": (chunk_agree is None
                             or chunk_agree >= SANITY_MIN)
                      and (d50_match is None
                           or d50_match >= SANITY_MIN)},
           "tau_sweep_descriptive": sweep,
           "verdicts": {"D1": [d1, round(dm_hit - ASM8_HIT, 4)],
                        "D2": [d2, round(dm_acc - ASM8_ACC, 4)],
                        "D3": [d3, round(dm_tok, 1)],
                        "D4": [d4, round(dh_acc - ASM8_ACC, 4)]},
           "records": recs}
    name = "pooljudge_smoke.json" if limit else "pooljudge.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("by_arm", "sanity", "tau_sweep_descriptive",
                       "verdicts")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
