"""H-series: abstention vs the rank-1 fallback (protocol.md,
registered pre-data). Committed-data-only amendment (Z-series
precedent): re-decides the committed D-H fallback questions with a
calibrated floor — abstain iff max pool cal p < phi (primary .15,
chosen a priori from the public diagnostic). Zero model calls.

    python abstain.py --smoke   # 2 questions, never scored
    python abstain.py           # scored run (offline; no slot needed)
"""
import argparse
import json
import os

import numpy as np

import asm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PHI = 0.15
SWEEP_PHI = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
HIGH_TAU = 0.70


def run(limit=0):
    with open(os.path.join(RES, "pooljudge.json")) as f:
        pj = json.load(f)
    recs_all = pj["records"]
    recs = recs_all[:limit] if limit else recs_all
    DH_ACC = pj["by_arm"]["dh"]["ans_acc"]

    # recompute committed D-H decisions from the committed s_vecs
    derived = []
    for r in recs:
        cal = [asm.cal_p(v["p"]) if v["p"] is not None else None
               for v in r["s_vec"]]
        stop = next((v["ci"] for v, d in zip(r["s_vec"], cal)
                     if d is not None and d >= HIGH_TAU), None)
        assert stop == r["dh"]["stop_ci"], f"decision drift at {r['q']}"
        assert (stop is None) == bool(r["dh"]["fallback"])
        derived.append({"r": r, "cal": cal, "stop": stop,
                        "maxp": max((d for d in cal if d is not None),
                                    default=0.0)})

    def arm(phi):
        """Returns answered acc-given-answered, coverage, abstention
        precision, n_abstained. Stopped questions always answer;
        fallback questions answer iff maxp >= phi."""
        ok, cov, abst, abst_wrong = 0, 0, 0, 0
        for d in derived:
            answered = d["stop"] is not None or d["maxp"] >= phi
            if answered:
                ok += d["r"]["dh"]["ans_ok"]
                cov += 1
            else:
                abst += 1
                abst_wrong += 1 - d["r"]["dh"]["ans_ok"]
        n = len(derived)
        return {"acc_given": ok / cov if cov else None,
                "coverage": cov / n,
                "abstention_precision": (abst_wrong / abst if abst
                                         else None),
                "n_abstained": abst}

    primary = arm(PHI)
    sweep = []
    for phi in SWEEP_PHI:
        a = arm(phi)
        sweep.append({"phi": phi, **{k: (round(v, 4) if isinstance(v, float)
                                          else v)
                                     for k, v in a.items()}})

    h1_val = (primary["acc_given"] - DH_ACC
              if primary["acc_given"] is not None else -1.0)
    h1 = ("PASS" if h1_val >= 0 else
          "PARTIAL" if h1_val >= -0.02 else "FAIL")
    prec = primary["abstention_precision"]
    h2 = ("PASS" if prec is not None and prec >= 0.70 else
          "PARTIAL" if prec is not None and prec >= 0.50 else "FAIL")
    h3 = "PASS-budget"  # zero model calls + asserted 58/58 reproduction

    out = {"tag": "abstain", "smoke": bool(limit), "limit": limit or 0,
           "commit": asm.CR.git_commit(),
           "committed_anchors": {"dh_acc": DH_ACC, "high_tau": HIGH_TAU},
           "phi_primary": PHI, "sweep_phi": list(SWEEP_PHI),
           "llm_calls": 0,
           "by_arm": {"dh_committed": {
               "ans_acc": DH_ACC,
               "coverage": 1.0,
               "fallback_ans_ok": (float(np.mean(
                   [d["r"]["dh"]["ans_ok"] for d in derived
                    if d["stop"] is None])) if any(d["stop"] is None
                                                   for d in derived)
                   else None),
               "stopped_ans_ok": (float(np.mean(
                   [d["r"]["dh"]["ans_ok"] for d in derived
                    if d["stop"] is not None])) if any(d["stop"]
                                                       is not None
                                                       for d in derived)
                   else None)},
               "hc_floor": {k: (round(v, 4) if isinstance(v, float)
                                else v) for k, v in primary.items()}},
           "phi_sweep_descriptive": sweep,
           "sanity": {"decision_reproduction": 1.0, "ok": True},
           "verdicts": {"H1": [h1, round(h1_val, 4)],
                        "H2": [h2, (round(prec, 4) if prec is not None
                                    else None)],
                        "H3": [h3, 0]}}
    name = "abstain_smoke.json" if limit else "abstain.json"
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("by_arm", "phi_sweep_descriptive", "verdicts")},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(2 if a.smoke else 0)
