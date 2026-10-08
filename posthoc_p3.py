"""POST-HOC (disclosed in protocol RESULTS): leaf-level conditional accuracy
at m in {0.0, 0.5} — the registered P3 metric read L3, which the dial does
not mark; the 50% sibling floor lives at L4. Deterministic rerun, seed 0."""
import run_lab as R, heads, walks
for m in (0.0, 0.5):
    Xtr, ytr, Xev, yev = R.prep(m)
    Xp, yp = R.subsample(Xtr, ytr, 256, 0)
    for arm in ("B", "C"):
        mdl = heads.train(arm, Xp, yp, seed=0)
        w4 = walks.hard_walk(mdl, Xev, yev, 4)
        print(f"POSTHOC m={m} {arm}: acc_L4 (leaf|parents correct) = "
              f"{w4['acc_L4']:.4f}  path_d4 = {w4['path_acc']:.4f}", flush=True)
