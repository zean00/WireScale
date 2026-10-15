"""F-series: doc-selection (protocol.md, registered pre-data).

Every scored machine so far got the gold doc for free — cross-document
ranking has never been measured. Rank the 4 docs by MAX chunk cosine,
then run the scored machine (top-8 pool, S verifier, tau=.70) inside
the selected doc (F-DOC) or inside a global top-8 pool with no doc
layer at all (F-GLOB). Anchor: committed D-H .4483 @ 1,422.5 tok/q.

    python fselect.py --smoke   # 2 questions, never scored
    python fselect.py           # scored run (needs slot 7; no tunnel)
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
HIGH_TAU = 0.70
POOL_N = 8
PF4_CEILING = 1800.0
SANITY_MIN = 0.95

JUDGE_USER = ("Question: {q}\n\nPassage:\n{c}\n\n"
              "Does the passage contain information that helps answer "
              "the question?")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def judge_t(q, chunk):
    """S instrument: single temp-0 judge call, CR prompt verbatim."""
    r = CR.llm_chat([{"role": "system", "content": CR.JUDGE_SYS},
                     {"role": "user", "content":
                      JUDGE_USER.format(q=q, c=chunk)}],
                    temp=0, max_tokens=CR.LLM_MAXT, logprobs=True)
    p = CR.p_yes_of(r)
    return p, r.get("usage", {})


def answer_f(q, gen_ch, gen_ci, gold_ch, gold_ci, seed, hit):
    """asm.answer_arm with separate generate/grade sources (pools may
    cross docs; grading always uses the question's gold doc)."""
    ans, gu = asm.gen_answer(q, gen_ch[gen_ci], seed)
    ok, gp, gru = asm.grade(q, ans, gold_ch[gold_ci])
    tok = gu.get("prompt_tokens", 0) + gru.get("prompt_tokens", 0)
    return {"hit": int(hit), "ans_ok": int(ok),
            "grade_p": (round(float(gp), 4) if gp is not None else None),
            "ans": ans[:200], "tok": tok}


def scan_pool(q, ch_map, pool, wr_pyes, gold_doc):
    """Judge in pool order; stop at cal p >= HIGH_TAU. Returns
    (stop_(doc,ci) or None, n_judged, tok, agreements)."""
    tok, agrs = 0, []
    for n, (doc, ci) in enumerate(pool):
        p, u = judge_t(q, ch_map[doc][ci])
        tok += u.get("prompt_tokens", 0)
        d = asm.cal_p(p)
        stop = bool((d >= HIGH_TAU) if d is not None else False)
        ref = wr_pyes.get(str(ci)) if doc == gold_doc else None
        if ref is not None:
            agrs.append(int(stop == (ref >= 0.5)))
        if stop:
            return (doc, ci), n + 1, tok, agrs
    return None, len(pool), tok, agrs


def top_pool(xcache, ch_map, qv, doc=None, n=POOL_N):
    """Top-n (doc, ci) by cosine; doc=None ranks across all docs.
    Stable sort over (doc order, chunk index) — matches the committed
    within-doc pool rule when doc is the gold doc."""
    cand = ((d, ci) for d in ([doc] if doc else ch_map)
            for ci in range(len(ch_map[d])))
    return sorted(cand, key=lambda dc: -float(
        xcache[ch_map[dc[0]][dc[1]]] @ qv))[:n]


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    with open(os.path.join(RES, "topk.json")) as f:
        committed_picks = {(r["doc"], r["q"]): r["topk8"]["pick"]
                           for r in json.load(f)["records"]}
    with open(os.path.join(RES, "pooljudge.json")) as f:
        pj = json.load(f)
    with open(os.path.join(RES, "walkret.json")) as f:
        wr = json.load(f)
    wrmap = {(r["doc"], r["q"]): r["p_yes"] for r in wr["records"]}
    dh_recs = {(r["doc"], r["q"]): r["dh"] for r in pj["records"]}
    DH_ACC = pj["by_arm"]["dh"]["ans_acc"]
    xcache = asm.load_ext_cache()
    evals = qd["eval"][:limit] if limit else qd["eval"]

    recs = []
    for i, e in enumerate(evals):
        gold_doc, gold, q = e["doc"], e["chunk"], e["q"]
        gold_ch = cm[gold_doc]
        qv = xcache[q]
        seed = asm.GEN_SEED_BASE + 17 * i

        dscore = {d: max(float(xcache[c] @ qv) for c in cm[d]) for d in cm}
        sel = max(dscore, key=dscore.get)   # dict order = doc order tie-break
        doc_pool = top_pool(xcache, cm, qv, doc=sel)
        glob_pool = top_pool(xcache, cm, qv)

        s_doc, n_doc, tok_doc, agr_doc = scan_pool(
            q, cm, doc_pool, wrmap.get((gold_doc, q), {}), gold_doc)
        s_glb, n_glb, tok_glb, agr_glb = scan_pool(
            q, cm, glob_pool, wrmap.get((gold_doc, q), {}), gold_doc)

        ci_doc = s_doc if s_doc is not None else doc_pool[0]
        ci_glb = s_glb if s_glb is not None else glob_pool[0]
        a_doc = answer_f(q, cm[ci_doc[0]], ci_doc[1], gold_ch, gold, seed,
                         hit=(ci_doc == (gold_doc, gold)))
        a_glb = answer_f(q, cm[ci_glb[0]], ci_glb[1], gold_ch, gold, seed,
                         hit=(ci_glb == (gold_doc, gold)))

        dh = dh_recs[(gold_doc, q)]
        on_gold = sel == gold_doc
        recs.append({
            "doc": gold_doc, "recipe": e["recipe"], "chunk": gold, "q": q,
            "sel_doc": sel, "doc_hit": int(on_gold),
            "doc_scores": {d: round(v, 4) for d, v in dscore.items()},
            "doc_pool": [[d, ci] for d, ci in doc_pool],
            "glob_pool": [[d, ci] for d, ci in glob_pool],
            "pool_matches_committed": (
                [ci for _, ci in doc_pool] == committed_picks[(gold_doc, q)]
                if on_gold else None),
            "fd": {**a_doc, "ci": list(ci_doc),
                   "stop_ci": (list(s_doc) if s_doc else None),
                   "fallback": int(s_doc is None), "n_judged": n_doc,
                   "judge_tok": tok_doc,
                   "matches_dh": int(on_gold and ci_doc[1] == dh["ci"]
                                     and a_doc["ans"] == dh["ans"])},
            "fg": {**a_glb, "ci": list(ci_glb),
                   "stop_ci": (list(s_glb) if s_glb else None),
                   "fallback": int(s_glb is None), "n_judged": n_glb,
                   "judge_tok": tok_glb},
            "sanity_agree": (float(np.mean(agr_doc + agr_glb))
                             if (agr_doc + agr_glb) else None)})
        r = recs[-1]
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"sel={r['sel_doc']:12s} hit={r['doc_hit']} | "
            f"FD hit={r['fd']['hit']} ok={r['fd']['ans_ok']} "
            f"fb={r['fd']['fallback']} dhsame={r['fd']['matches_dh']} | "
            f"FG hit={r['fg']['hit']} ok={r['fg']['ans_ok']} "
            f"fb={r['fg']['fallback']}")

    def m(arm, key, subset=None):
        rs = [r[arm][key] for r in recs
              if subset is None or r["recipe"] == subset]
        return float(np.mean(rs)) if rs else None

    doc_hit = float(np.mean([r["doc_hit"] for r in recs]))
    fd_acc, fg_acc = m("fd", "ans_ok"), m("fg", "ans_ok")
    fd_tok = m("fd", "judge_tok") + m("fd", "tok")
    p_ok = [r["pool_matches_committed"] for r in recs
            if r["pool_matches_committed"] is not None]
    pool_agree = float(np.mean(p_ok)) if p_ok else None
    dh_same = [r["fd"]["matches_dh"] for r in recs if r["doc_hit"] == 1]
    dh_same = float(np.mean(dh_same)) if dh_same else None
    agrs = [r["sanity_agree"] for r in recs if r["sanity_agree"] is not None]
    agree = float(np.mean(agrs)) if agrs else None

    pf1 = ("PASS" if doc_hit >= 0.95 else
           "PARTIAL" if doc_hit >= 0.90 else "FAIL")
    d2 = fd_acc - DH_ACC
    pf2 = ("PASS" if d2 >= -0.02 else "PARTIAL" if d2 >= -0.04 else "FAIL")
    d3 = fg_acc - fd_acc
    pf3 = ("PASS" if d3 >= -0.02 else "PARTIAL" if d3 >= -0.04 else "FAIL")
    pf4 = ("PASS-budget" if fd_tok <= PF4_CEILING else "FAIL-budget")
    sanity_ok = ((pool_agree is None or pool_agree >= SANITY_MIN)
                 and (dh_same is None or dh_same >= SANITY_MIN)
                 and (agree is None or agree >= SANITY_MIN))
    out = {"tag": "fselect", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "tau": HIGH_TAU, "pool_n": POOL_N,
           "committed_anchors": {"dh_acc": DH_ACC,
                                 "pf4_ceiling_tok": PF4_CEILING},
           "by_arm": {
               "selection": {"doc_hit": doc_hit,
                             "per_recipe": {rc: float(np.mean(
                                 [r["doc_hit"] for r in recs
                                  if r["recipe"] == rc]))
                                 for rc in ("direct", "indirect")}},
               "fd": {"ans_acc": fd_acc, "stop_hit": m("fd", "hit"),
                      "fallback_rate": m("fd", "fallback"),
                      "n_judged": m("fd", "n_judged"),
                      "tok_per_q": fd_tok,
                      "per_recipe": {rc: m("fd", "ans_ok", rc)
                                     for rc in ("direct", "indirect")}},
               "fg": {"ans_acc": fg_acc, "stop_hit": m("fg", "hit"),
                      "fallback_rate": m("fg", "fallback"),
                      "n_judged": m("fg", "n_judged"),
                      "tok_per_q": m("fg", "judge_tok") + m("fg", "tok"),
                      "per_recipe": {rc: m("fg", "ans_ok", rc)
                                     for rc in ("direct", "indirect")}}},
           "sanity": {"pool_matches_committed": pool_agree,
                      "dh_answer_identity_on_hit": dh_same,
                      "judge_agree": agree, "ok": sanity_ok},
           "verdicts": {"PF1": [pf1, round(doc_hit, 4)],
                        "PF2": [pf2, round(d2, 4)],
                        "PF3": [pf3, round(d3, 4)],
                        "PF4": [pf4, round(fd_tok, 1)]},
           "records": recs}
    name = "fselect_smoke.json" if limit else "fselect.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("by_arm", "sanity", "verdicts")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
