"""Z-series: temperature calibration of judge confidence (registered).

Uses ONLY committed data (walkret.json flat per-pair p_yes; disclosed
in the registration). Seeded 29/29 question split; fit one temperature
on the logit, evaluate ECE raw vs calibrated on holdout.

    python calib.py    # no server, seconds
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SPLIT_SEED, N_BINS = 20261014, 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ece(p, y, bins=N_BINS):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1
                              else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def nll(p, y, T):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    q = 1 / (1 + np.exp(-z / T))
    return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def main():
    wr = json.load(open(os.path.join(RES, "walkret.json")))
    pairs = []  # (qidx, p, y)
    for qi, r in enumerate(wr["records"]):
        for ci, p in r["p_yes"].items():
            if p is None:
                continue
            pairs.append((qi, float(p), int(int(ci) == r["chunk"])))
    qs = sorted({q for q, _, _ in pairs})
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(qs))
    fit_q = {qs[i] for i in perm[:len(qs) // 2]}
    fit = [(p, y) for q, p, y in pairs if q in fit_q]
    hold = [(p, y) for q, p, y in pairs if q not in fit_q]
    log(f"pairs: fit {len(fit)}, holdout {len(hold)} "
        f"(questions {len(fit_q)}/{len(qs) - len(fit_q)})")

    Ts = np.linspace(0.2, 5.0, 481)
    nlls = [nll([p for p, _ in fit], [y for _, y in fit], T) for T in Ts]
    T = float(Ts[int(np.argmin(nlls))])

    def ev(slice_):
        p = np.asarray([x[0] for x in slice_])
        y = np.asarray([x[1] for x in slice_])
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6)
                   / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        pc = 1 / (1 + np.exp(-z / T))
        return ece(p, y), ece(pc, y)

    fit_raw, fit_cal = ev(fit)
    hold_raw, hold_cal = ev(hold)

    # early-stop bar transfer (descriptive): tau reaching >=90% of flat
    # recall within each slice, raw vs calibrated
    def tau90(slice_):
        p = np.asarray([x[0] for x in slice_])
        y = np.asarray([x[1] for x in slice_])
        if not (y == 1).any():
            return None
        target = 0.9 * float(y.mean())
        for t in np.linspace(0.05, 0.95, 19):
            if float((p >= t)[y == 1].mean()) >= target:
                return round(float(t), 2)
        return None

    za = ("PASS" if hold_cal < hold_raw - 0.01 else
          "PARTIAL" if abs(hold_cal - hold_raw) <= 0.01 else "FAIL")
    zb = ("PASS" if 0.5 <= T <= 2.0
          and (fit_cal < fit_raw) == (hold_cal < hold_raw) else "FAIL")
    out = {"tag": "calib", "smoke": False, "T": round(T, 3),
           "n_pairs": {"fit": len(fit), "holdout": len(hold)},
           "ece": {"fit": [round(fit_raw, 4), round(fit_cal, 4)],
                   "holdout": [round(hold_raw, 4), round(hold_cal, 4)]},
           "tau90_descriptive": {"fit": tau90(fit), "holdout": tau90(hold)},
           "verdicts": {"ZA": [za, round(hold_raw - hold_cal, 4)],
                        "ZB": zb}}
    path = os.path.join(RES, "calib.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {path}")
    print(json.dumps(out["verdicts"], indent=1))
    print(json.dumps({k: out[k] for k in ("T", "ece", "tau90_descriptive")},
                     indent=1))


if __name__ == "__main__":
    main()
