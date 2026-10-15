"""G-series: the grader audit (protocol.md, registered pre-data).

Every answer-level number rests on one grader = the frozen generator
with one prompt. This re-grades the committed answer STRINGS of the
four headline arms (ASM, TOP1, TOPK-2, ASM-8; n=58 each) under two
altered grading instruments: G1 (reworded grader) and G2 (3-sample
self-consistency on the original prompt). No regeneration.

    python regrade.py --smoke   # 2 questions, never scored
    python regrade.py           # scored run (needs slot 7; no tunnel)
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
G1_SYS = ("You are a careful grader. Reply with exactly one word: "
          "yes or no.")
G1_USER = ("Reference passage:\n{gold}\n\nQuestion: {q}\n\n"
           "Proposed answer: {ans}\n\nIs the proposed answer a correct "
           "answer to the question, according to the reference passage?")
N_SAMPLES = 3
SAMP_SEED_BASE = 20261016
ARMS = ("asm", "top1", "topk2", "asm8")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def g1(q, ans, gold_chunk):
    r = CR.llm_chat([{"role": "system", "content": G1_SYS},
                     {"role": "user", "content":
                      G1_USER.format(gold=gold_chunk, q=q, ans=ans)}],
                    temp=0, max_tokens=CR.LLM_MAXT, logprobs=True)
    p = CR.p_yes_of(r)
    content = r["choices"][0]["message"]["content"].strip().lower()
    ok = (p >= 0.5) if p is not None else \
        bool(content) and content.split()[0].startswith("y")
    return int(bool(ok)), p, r.get("usage", {})


def g2(q, ans, gold_chunk, seed):
    r = CR.llm_chat([{"role": "system", "content": asm.GRADE_SYS},
                     {"role": "user", "content":
                      f"Reference passage:\n{gold_chunk}\n\n"
                      f"Question: {q}\n\nProposed answer: {ans}\n\n"
                      "Does the proposed answer correctly answer the "
                      "question, as supported by the reference passage?"}],
                    temp=1.0, max_tokens=CR.LLM_MAXT, logprobs=True,
                    seed=seed)
    return CR.p_yes_of(r)


def run(limit=0):
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    with open(os.path.join(RES, "assembly.json")) as f:
        asmrecs = {(r["doc"], r["q"]): r for r in json.load(f)["records"]}
    with open(os.path.join(RES, "topk.json")) as f:
        topkrecs = {(r["doc"], r["q"]): r for r in json.load(f)["records"]}
    with open(os.path.join(RES, "poolstop.json")) as f:
        poolrecs = {(r["doc"], r["q"]): r for r in json.load(f)["records"]}
    goldmap = {(e["doc"], e["q"]): e["chunk"] for e in qd["eval"]}
    evals = qd["eval"][:limit] if limit else qd["eval"]

    recs = []
    for i, e in enumerate(evals):
        key = (e["doc"], e["q"])
        gold = goldmap[key]
        gold_chunk = cm[e["doc"]][gold]
        answers = {
            "asm": asmrecs[key]["asm"]["ans"],
            "top1": asmrecs[key]["top1"]["ans"],
            "topk2": topkrecs[key]["topk2"]["ans"],
            "asm8": poolrecs[key]["asm8"]["ans"],
        }
        committed = {
            "asm": asmrecs[key]["asm"]["ans_ok"],
            "top1": asmrecs[key]["top1"]["ans_ok"],
            "topk2": topkrecs[key]["topk2"]["ans_ok"],
            "asm8": poolrecs[key]["asm8"]["ans_ok"],
        }
        row = {"doc": e["doc"], "recipe": e["recipe"], "chunk": gold,
               "q": e["q"], "committed": committed}
        for arm, ans in answers.items():
            ok1, p1, u1 = g1(e["q"], ans, gold_chunk)
            ps = []
            for s in range(N_SAMPLES):
                p2 = g2(e["q"], ans, gold_chunk,
                        SAMP_SEED_BASE + 101 * i + 13 * s)
                if p2 is not None:
                    ps.append(p2)
            mean_p = float(np.mean(ps)) if ps else None
            ok2 = int(mean_p >= 0.5) if mean_p is not None else ok1
            row[arm] = {"g1_ok": ok1, "g1_p": (round(float(p1), 4)
                                               if p1 is not None else None),
                        "g2_ok": ok2, "g2_p": (round(mean_p, 4)
                                               if mean_p is not None
                                               else None)}
        recs.append(row)
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} " +
            " ".join(f"{a}:G1={row[a]['g1_ok']}/G2={row[a]['g2_ok']}"
                     f"(c{committed[a]})" for a in ARMS))

    def acc(arm, key, subset=None):
        rs = [r[arm][key] for r in recs
              if subset is None or r["recipe"] == subset]
        return float(np.mean(rs)) if rs else None

    summary = {}
    for inst, key in (("g1", "g1_ok"), ("g2", "g2_ok")):
        a = {arm: acc(arm, key) for arm in ARMS}
        c = {arm: acc(arm, "g1_p" if key == "g1_ok" else "g2_p")
             for arm in ARMS}
        flips = {"asm_vs_topk2": round(a["topk2"] - a["asm"], 4),
                 "asm_vs_top1": round(a["top1"] - a["asm"], 4)}
        ga = ("PASS" if flips["asm_vs_topk2"] <= 0
              and flips["asm_vs_top1"] <= 0 else
              "PARTIAL" if max(flips.values()) <= 0.02 else "FAIL")
        drift = {arm: round(abs(a[arm] - float(np.mean(
            [r["committed"][arm] for r in recs]))), 4) for arm in ARMS}
        gb = ("PASS" if max(drift.values()) <= 0.05 else "FAIL")
        summary[inst] = {"acc": {k: round(v, 4) for k, v in a.items()},
                         "mean_grade_p": {k: (round(v, 4) if v is not None
                                              else None)
                                          for k, v in c.items()},
                         "flips_vs_asm": flips,
                         "drift_vs_committed": drift,
                         "GA": ga, "GB": gb}
    out = {"tag": "regrade", "smoke": bool(limit), "limit": limit,
           "commit": CR.git_commit(), "model_fp": EC.fingerprint(),
           "n_samples_g2": N_SAMPLES,
           "committed_acc": {arm: round(float(np.mean(
               [r["committed"][arm] for r in recs])), 4) for arm in ARMS},
           "summary": summary,
           "records": recs}
    name = "regrade_smoke.json" if limit else "regrade.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("committed_acc", "summary")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
