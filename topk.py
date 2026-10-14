"""B-series: the fair fight — machine vs production-style top-k RAG
(protocol.md, registered pre-data).

A-series beat pure RAG at k=1, the strictest variant. Production
stacks feed top-k excerpts in one context, and committed E-series
numbers say top-3 context presence (.638) exceeds the machine's
answered-chunk hit (.466). Per frozen eval question (n=58):
TOPK-k for k in {2,3,5,8} — top-k chunks in E0 embedder order,
one prompt, generate + grade exactly as A-series — scored against
the committed ASM records (paired; same seeds). TOP1 is re-run as
the determinism sanity.

    python topk.py --smoke   # 2 questions, never scored
    python topk.py           # scored run (needs slot 7; no tunnel)
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
SWEEP_K = (2, 3, 5, 8)
ASM_TOK_COMMITTED = 5427.672413793103   # assembly.json by_arm.asm.tok_per_q
B_PARTIAL = -0.02
SANITY_MIN = 0.95


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_answer_k(q, chunks, seed):
    body = "\n\n".join(f"Excerpt {j}:\n{c}"
                       for j, c in enumerate(chunks, 1))
    r = CR.llm_chat([{"role": "system", "content": asm.GEN_SYS},
                     {"role": "user", "content": f"Question: {q}\n\n{body}"}],
                    temp=0, max_tokens=asm.GEN_MAXT, seed=seed)
    return (r["choices"][0]["message"]["content"].strip(),
            r.get("usage", {}))


def topk_arm(q, ch, pick, gold, seed):
    ans, gu = gen_answer_k(q, [ch[ci] for ci in pick], seed)
    ok, gp, gru = asm.grade(q, ans, ch[gold])
    return {"pick": list(pick), "k_eff": len(pick),
            "ctx_hit": int(gold in pick), "ans_ok": int(ok),
            "grade_p": (round(float(gp), 4) if gp is not None else None),
            "ans": ans[:200],
            "tok": gu.get("prompt_tokens", 0) + gru.get("prompt_tokens", 0)}


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    xcache = asm.load_ext_cache()
    with open(os.path.join(RES, "assembly.json")) as f:
        committed = json.load(f)
    crecs = {(r["doc"], r["q"]): r for r in committed["records"]}
    evals = qd["eval"][:limit] if limit else qd["eval"]

    recs = []
    for i, e in enumerate(evals):
        ch = cm[e["doc"]]
        gold, q = e["chunk"], e["q"]
        qv = xcache[q]
        emb_order = sorted(range(len(ch)),
                           key=lambda ci: -float(xcache[ch[ci]] @ qv))
        seed = asm.GEN_SEED_BASE + 17 * i
        rec = {"doc": e["doc"], "recipe": e["recipe"], "chunk": gold, "q": q}
        for k in SWEEP_K:
            pick = emb_order[:min(k, len(ch))]
            rec[f"topk{k}"] = topk_arm(q, ch, pick, gold, seed)
        # determinism sanity: A-series TOP1 arm, byte-identical code path
        t1_ans, t1u = asm.gen_answer(q, ch[emb_order[0]], seed)
        t1_ok, t1_gp, t1gru = asm.grade(q, t1_ans, ch[gold])
        rec["top1"] = {"ci": emb_order[0], "hit": int(emb_order[0] == gold),
                       "ans_ok": int(t1_ok),
                       "grade_p": (round(float(t1_gp), 4)
                                   if t1_gp is not None else None),
                       "ans": t1_ans[:200],
                       "tok": t1u.get("prompt_tokens", 0)
                       + t1gru.get("prompt_tokens", 0)}
        cref = crecs.get((e["doc"], q))
        rec["c_top1"] = cref["top1"] if cref else None
        rec["c_asm"] = cref["asm"] if cref else None
        recs.append(rec)
        agree = (int(rec["top1"]["ans"] == cref["top1"]["ans"]
                     and rec["top1"]["ci"] == cref["top1"]["ci"])
                 if cref else None)
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} " +
            " ".join(f"k{k}={rec[f'topk{k}']['ans_ok']}"
                     f"({rec[f'topk{k}']['ctx_hit']})"
                     for k in SWEEP_K) +
            f" | t1_agree={agree if agree is None else bool(agree)}")

    def m(arm, key, subset=None):
        rs = [r[arm][key] for r in recs
              if subset is None or r["recipe"] == subset]
        return float(np.mean(rs)) if rs else None

    by_k = {}
    for k in SWEEP_K:
        by_k[k] = {"ans_acc": m(f"topk{k}", "ans_ok"),
                   "ctx_hit": m(f"topk{k}", "ctx_hit"),
                   "tok_per_q": m(f"topk{k}", "tok"),
                   "per_recipe": {rc: m(f"topk{k}", "ans_ok", rc)
                                  for rc in ("direct", "indirect")}}
    asm_acc = committed["by_arm"]["asm"]["ans_acc"]

    ag = [int(r["top1"]["ans"] == r["c_top1"]["ans"]
              and r["top1"]["ci"] == r["c_top1"]["ci"])
          for r in recs if r["c_top1"]]
    sanity = float(np.mean(ag)) if ag else None
    t1_acc_drift = (abs(m("top1", "ans_ok")
                        - float(np.mean([r["c_top1"]["ans_ok"]
                                         for r in recs if r["c_top1"]])))
                    if ag else None)

    S = [k for k in SWEEP_K if by_k[k]["tok_per_q"] <= ASM_TOK_COMMITTED]
    d1 = asm_acc - by_k[3]["ans_acc"]
    b1 = ("PASS" if d1 >= 0 else
          "PARTIAL" if d1 >= B_PARTIAL else "FAIL")
    if S:
        dB = asm_acc - max(by_k[k]["ans_acc"] for k in S)
        b2 = ("PASS" if dB >= 0 else
              "PARTIAL" if dB >= B_PARTIAL else "FAIL")
    else:
        dB, b2 = None, "n/a"
    out = {"tag": "topk", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "sweep_k": list(SWEEP_K),
           "committed_anchors": {"asm_acc": asm_acc,
                                 "asm_tok": ASM_TOK_COMMITTED},
           "by_k": by_k,
           "asm_committed": committed["by_arm"]["asm"],
           "sanity": {"top1_agree": sanity,
                      "top1_acc_drift": (round(t1_acc_drift, 4)
                                         if t1_acc_drift is not None
                                         else None),
                      "ok": sanity is None or sanity >= SANITY_MIN},
           "verdicts": {"B1": [b1, round(d1, 4)],
                        "B2": [b2, (round(dB, 4) if dB is not None
                                    else None)],
                        "budget_set_S": S},
           "records": recs}
    name = "topk_smoke.json" if limit else "topk.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("by_k", "sanity", "verdicts")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
