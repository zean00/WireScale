"""W-series: retrieval walk vs flat (protocol.md, registered pre-data).

The founding question in the one regime where the read is EXPENSIVE:
exhaustive J-llm judging costs ~29k prompt tokens / ~17 s per
question. WALK = menu pick of one 20-chunk window, then judge inside
it. FLAT = exhaustive judge (fresh same-session rerun; sanity band
vs R run 2). Per-pair p_yes scores are persisted.

    python walkret.py --smoke   # plumbing check, never scored
    python walkret.py           # scored run
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import chunkrel as CR

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WINDOW_N = 20
PREV_CHARS = 200
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def windows(n):
    return [list(range(s, min(s + WINDOW_N, n)))
            for s in range(0, n, WINDOW_N)]


def window_previews(doc, cm):
    ch = cm[doc]
    ws = windows(len(ch))
    prev = [f"chunks {w[0] + 1}-{w[-1] + 1}: "
            f"{ch[w[0]].replace(chr(10), ' ')[:PREV_CHARS]}" for w in ws]
    return ws, prev


def run_eval(cm, evals, r2rec):
    records = []
    for i, e in enumerate(evals):
        ch, gold = cm[e["doc"]], e["chunk"]
        # FLAT: exhaustive judge
        t0 = time.time()
        yes, ps, ftok = [], {}, 0
        for ci, ct in enumerate(ch):
            y, p, u = CR.judge(e["q"], ct)
            ftok += u.get("prompt_tokens", 0)
            if p is not None:
                ps[str(ci)] = round(float(p), 4)
            if y:
                yes.append(ci)
        flat_ms = (time.time() - t0) * 1000
        # WALK: window menu pick, then judge inside
        ws, prev = window_previews(e["doc"], cm)
        wtok, w_ms, single = 0, 0.0, len(ws) == 1
        if single:
            pick = 1
        else:
            t0 = time.time()
            pick, u = CR.menu_pick(e["q"], prev)
            wtok += u.get("prompt_tokens", 0)
            w_ms += (time.time() - t0) * 1000
        abstain, wyes = pick == 0, []
        if not abstain:
            t0 = time.time()
            for ci in ws[pick - 1]:
                y, p, u = CR.judge(e["q"], ch[ci])
                wtok += u.get("prompt_tokens", 0)
                if y:
                    wyes.append(ci)
            w_ms += (time.time() - t0) * 1000
        rec = {"doc": e["doc"], "chunk": gold, "recipe": e["recipe"],
               "q": e["q"],
               "flat_recall": int(gold in yes), "flat_nyes": len(yes),
               "flat_tok": ftok, "flat_ms": round(flat_ms),
               "walk_recall": int(gold in wyes), "walk_nyes": len(wyes),
               "walk_tok": wtok, "walk_ms": round(w_ms),
               "single_window": single, "abstain": abstain,
               "pick": pick,
               "win_correct": (None if single or abstain
                               else int(gold in ws[pick - 1])),
               "p_yes": ps}
        if r2rec is not None:
            rec["r2_jllm"] = r2rec[i]["jllm_recall"]
        records.append(rec)
        log(f"  q{i + 1}/{len(evals)} {e['recipe']:8s} "
            f"flat={rec['flat_recall']} walk={rec['walk_recall']} "
            f"pick={pick}{'/single' if single else ''} "
            f"winyes={len(wyes)}")
    return records


def summarize(records):
    n = len(records)

    def m(key, subset=None):
        rs = [r for r in records
              if subset is None or r["recipe"] == subset]
        return float(np.mean([r[key] for r in rs])) if rs else None

    by = {arm: {rec: m(f"{arm}_recall", rec)
                for rec in ("direct", "indirect")}
          for arm in ("flat", "walk")}
    for arm in ("flat", "walk"):
        by[arm]["all"] = m(f"{arm}_recall")
    multi = [r for r in records if not r["single_window"]]
    picked = [r for r in records if not r["abstain"]]
    p_win = (float(np.mean([r["win_correct"] for r in multi
                            if r["win_correct"] is not None]))
             if any(r["win_correct"] is not None for r in multi) else None)
    p_yes_g_win = (float(np.mean([r["walk_recall"] for r in picked]))
                   if picked else None)
    tok_f = m("flat_tok")
    tok_w = m("walk_tok")
    sweep = []
    for t in THRESHOLDS:
        rec_t = float(np.mean([int(r["p_yes"].get(str(r["chunk"]), 0) >= t)
                               for r in records]))
        size_t = float(np.mean([sum(1 for v in r["p_yes"].values() if v >= t)
                                for r in records]))
        sweep.append({"t": t, "recall": round(rec_t, 4),
                      "mean_size": round(size_t, 2)})
    agree = (float(np.mean([int(r["flat_recall"] == r["r2_jllm"])
                            for r in records]))
             if records and "r2_jllm" in records[0] else None)
    return by, p_win, p_yes_g_win, tok_f, tok_w, sweep, agree


def run_smoke():
    global WINDOW_N
    WINDOW_N = 4
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    evals = [qd["eval"][0], next(e for e in qd["eval"]
                                 if e["doc"] == "protocol.md")]
    # smoke: judge only the first 12 chunks of each doc (plumbing)
    cm_smoke = {e["doc"]: cm[e["doc"]][:12] for e in evals}
    for e in evals:
        e["chunk"] = min(e["chunk"], 3)
    records = run_eval(cm_smoke, evals, None)
    with open(os.path.join(RES, "walkret_smoke.json"), "w") as f:
        json.dump({"tag": "walkret", "smoke": True, "window_n": WINDOW_N,
                   "records": records}, f, indent=1, sort_keys=True)
    log("smoke done (marked, never scored)")


def run_scored():
    if not os.path.exists(CR.QFILE):
        sys.exit("frozen question file missing")
    with open(os.path.join(RES, "chunkrel.json")) as f:
        r2rec = json.load(f)["records"]
    qd = json.load(open(CR.QFILE))
    cm = CR.build_chunkmap()
    log("windows: " + str({d: len(windows(len(ch)))
                           for d, ch in cm.items()}))
    records = run_eval(cm, qd["eval"], r2rec)
    by, p_win, p_yes_g_win, tok_f, tok_w, sweep, agree = summarize(records)

    sanity_ok = agree is None or agree >= 0.75
    d = by["flat"]["all"] - by["walk"]["all"]
    w1 = ("PASS" if d >= 0 else
          "PARTIAL" if d >= -0.02 else "FAIL")
    w2 = ("PASS" if (by["walk"]["all"] >= 0.8 * by["flat"]["all"]
                     and tok_w <= 0.35 * tok_f) else "FAIL")
    w3 = ("PASS" if p_yes_g_win is not None
          and p_yes_g_win >= 0.9 * by["flat"]["all"] else "FAIL")

    out = {"tag": "walkret", "smoke": False, "commit": CR.git_commit(),
           "window_n": WINDOW_N, "by_arm": by,
           "p_window": p_win, "p_yes_given_window": p_yes_g_win,
           "tokens": {"flat": tok_f, "walk": tok_w,
                      "ratio": round(tok_w / tok_f, 4)},
           "sanity": {"r2_agreement": agree, "ok": sanity_ok},
           "threshold_sweep_posthoc": sweep,
           "records": records,
           "verdicts": {"W1": w1, "flat_minus_walk": round(d, 4),
                        "W2": w2, "W3": w3}}
    out["ms"] = {"flat": float(np.mean([r["flat_ms"] for r in records])),
                 "walk": float(np.mean([r["walk_ms"] for r in records]))}
    path = os.path.join(RES, "walkret.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps({"by_arm": by, "p_window": p_win,
                      "p_yes_given_window": p_yes_g_win,
                      "tokens": out["tokens"], "ms": out["ms"],
                      "sanity": out["sanity"],
                      "verdicts": out["verdicts"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run_smoke() if a.smoke else run_scored()
