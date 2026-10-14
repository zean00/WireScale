"""A-series: end-to-end assembly (protocol.md, registered pre-data).

The endorsed recipe — rank with the dedicated embedder (E), scan-and-
stop with the judge as verifier (X), calibrated bar carried from Z —
has never been scored as one machine. Three arms per frozen eval
question (n=58): TOP1 (pure RAG from embedder top-1), FLATPOL (the
X-DS policy in doc order), ASM (embedder order + fallback-to-top1).
Answers are generated from the answered chunk and graded by the
frozen judge against the gold chunk.

    python asm.py --smoke   # 2 questions, never scored
    python asm.py           # scored run (needs slot 7; no tunnel)
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
TEMP_T = 0.66                       # committed calib.json
DS_STOP_COMMITTED = 0.3103448275862069   # xstop.json ds.stop
FLAT_TOK_COMMITTED = 29236.0
A4_TOK_RATIO = 0.35
SANITY_MIN = 0.90
GEN_MAXT = 128
GEN_SEED_BASE = 20261014

GEN_SYS = "Answer using only the excerpt. One sentence, no preamble."
GRADE_SYS = ("You are a strict grader. Answer with exactly one word: "
             "yes or no.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ext_cache():
    import extembed as EE
    with open(os.path.join(RES, "extembed.json")) as f:
        fp = json.load(f)["ext_fp"]
    return EE.load_ext_cache(fp)


def cal_p(p):
    if p is None:
        return None
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return float(1 / (1 + np.exp(-z / TEMP_T)))


def gen_answer(q, chunk, seed):
    r = CR.llm_chat([{"role": "system", "content": GEN_SYS},
                     {"role": "user", "content":
                      f"Question: {q}\n\nExcerpt:\n{chunk}"}],
                    temp=0, max_tokens=GEN_MAXT, seed=seed)
    return (r["choices"][0]["message"]["content"].strip(),
            r.get("usage", {}))


def grade(q, ans, gold_chunk):
    r = CR.llm_chat([{"role": "system", "content": GRADE_SYS},
                     {"role": "user", "content":
                      f"Reference passage:\n{gold_chunk}\n\n"
                      f"Question: {q}\n\nProposed answer: {ans}\n\n"
                      "Does the proposed answer correctly answer the "
                      "question, as supported by the reference passage?"}],
                    temp=0, max_tokens=CR.LLM_MAXT, logprobs=True)
    p = CR.p_yes_of(r)
    content = r["choices"][0]["message"]["content"].strip().lower()
    ok = (p >= 0.5) if p is not None else \
        bool(content) and content.split()[0].startswith("y")
    return bool(ok), p, r.get("usage", {})


def scan(q, ch, order, ref_pyes, use_cal):
    """Judge in order; stop at first p >= TAU (calibrated if use_cal).
    Returns (stopped_ci or None, n_judged, tok, agreements)."""
    tok, agreements = 0, []
    for n, ci in enumerate(order):
        yes, p, usage = CR.judge(q, ch[ci])
        tok += usage.get("prompt_tokens", 0)
        d = cal_p(p) if use_cal else p
        stop = bool((d >= TAU) if d is not None else yes)
        ref = ref_pyes.get(str(ci))
        if ref is not None:
            agreements.append(int(stop == (ref >= TAU)))
        if stop:
            return ci, n + 1, tok, agreements
    return None, len(order), tok, agreements


def answer_arm(q, ch, ci, gold, seed):
    ans, gu = gen_answer(q, ch[ci], seed)
    ok, gp, gru = grade(q, ans, ch[gold])
    tok = gu.get("prompt_tokens", 0) + gru.get("prompt_tokens", 0)
    return {"ci": ci, "hit": int(ci == gold), "ans_ok": int(ok),
            "grade_p": (round(float(gp), 4) if gp is not None else None),
            "ans": ans[:200], "tok": tok}


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
        gold, q = e["chunk"], e["q"]
        qv = xcache[q]
        emb_order = sorted(range(len(ch)),
                           key=lambda ci: -float(xcache[ch[ci]] @ qv))
        ref = next((r for r in wr["records"]
                    if r["doc"] == e["doc"] and r["q"] == q), None)
        ref_pyes = ref["p_yes"] if ref else {}
        seed = GEN_SEED_BASE + 17 * i

        top = answer_arm(q, ch, emb_order[0], gold, seed)
        s_ci, s_n, s_tok, s_agr = scan(q, ch, emb_order, ref_pyes, True)
        asm_fallback = s_ci is None
        asm = answer_arm(q, ch, emb_order[0] if asm_fallback else s_ci,
                         gold, seed)
        d_ci, d_n, d_tok, d_agr = scan(q, ch, list(range(len(ch))),
                                       ref_pyes, False)
        flp = answer_arm(q, ch, 0 if d_ci is None else d_ci, gold, seed)

        agrees = s_agr + d_agr
        recs.append({
            "doc": e["doc"], "recipe": e["recipe"], "chunk": gold, "q": q,
            "top1": top,
            "asm": {**asm, "fallback": int(asm_fallback),
                    "scan_stop_ci": s_ci, "n_judged": s_n,
                    "total_tok": s_tok + asm["tok"]},
            "flatpol": {**flp, "fallback": int(d_ci is None),
                        "scan_stop_ci": d_ci, "n_judged": d_n,
                        "total_tok": d_tok + flp["tok"]},
            "sanity_agree": (float(np.mean(agrees)) if agrees else None)})
        sa = recs[-1]["sanity_agree"]
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"TOP1 hit={top['hit']} ok={top['ans_ok']} | "
            f"ASM hit={asm['hit']} ok={asm['ans_ok']} "
            f"fb={int(asm_fallback)} n={s_n} | "
            f"FLP hit={flp['hit']} ok={flp['ans_ok']} n={d_n} | "
            f"agree={sa if sa is None else round(sa, 2)}")

    def m(arm, key, subset=None):
        rs = [r[arm][key] for r in recs
              if subset is None or r["recipe"] == subset]
        return float(np.mean(rs)) if rs else None

    top1_acc, asm_acc, flp_acc = (m("top1", "ans_ok"), m("asm", "ans_ok"),
                                  m("flatpol", "ans_ok"))
    asm_hit, asm_tok = m("asm", "hit"), m("asm", "total_tok")
    fb_rate = m("asm", "fallback")
    agrees = [r["sanity_agree"] for r in recs if r["sanity_agree"] is not None]
    sanity = float(np.mean(agrees)) if agrees else None

    d1 = asm_acc - top1_acc
    a1 = ("PASS" if d1 >= 0.03 else
          "PARTIAL" if abs(d1) <= 0.03 else "FAIL")
    d2 = asm_acc - flp_acc
    a2 = ("PASS" if d2 >= 0 else
          "PARTIAL" if d2 >= -0.02 else "FAIL")
    a3 = "PASS" if asm_hit >= DS_STOP_COMMITTED else "FAIL"
    a4 = ("PASS-budget" if asm_tok <= A4_TOK_RATIO * FLAT_TOK_COMMITTED
          else "FAIL-budget")
    out = {"tag": "assembly", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "tau": TAU, "temp_T": TEMP_T,
           "committed_anchors": {
               "xstop_ds_stop": DS_STOP_COMMITTED,
               "flat_tok": FLAT_TOK_COMMITTED,
               "a4_ceiling_tok": A4_TOK_RATIO * FLAT_TOK_COMMITTED},
           "by_arm": {
               "top1": {"ans_acc": top1_acc,
                        "per_recipe": {rc: m("top1", "ans_ok", rc)
                                       for rc in ("direct", "indirect")}},
               "asm": {"ans_acc": asm_acc, "stop_hit": asm_hit,
                       "fallback_rate": fb_rate, "tok_per_q": asm_tok,
                       "per_recipe": {rc: m("asm", "ans_ok", rc)
                                      for rc in ("direct", "indirect")}},
               "flatpol": {"ans_acc": flp_acc,
                           "tok_per_q": m("flatpol", "total_tok"),
                           "per_recipe": {rc: m("flatpol", "ans_ok", rc)
                                          for rc in ("direct", "indirect")}}},
           "sanity_agree": sanity,
           "sanity_ok": sanity is None or sanity >= SANITY_MIN,
           "verdicts": {"A1": [a1, round(d1, 4)],
                        "A2": [a2, round(d2, 4)],
                        "A3": [a3, round(asm_hit - DS_STOP_COMMITTED, 4)],
                        "A4": [a4, round(asm_tok, 1)]},
           "records": recs}
    name = "assembly_smoke.json" if limit else "assembly.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("by_arm", "sanity_agree", "sanity_ok", "verdicts")},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
