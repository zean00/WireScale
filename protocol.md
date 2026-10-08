# Tree-lab: does parent-conditioning make multi-level decision trees scale?

Registered 2026-10-07, before any data. Standalone testbed (deliberately
beyond wiresense): synthetic utterance-marked hierarchy, frozen real
encoder, controlled pool size / depth / markedness. Local commits only,
never pushed.

## Question

In a frozen-embedding + light-readout router, does conditioning child
readouts on the parent decision (parent one-hot as a vector feature)
flatten the depth-slope of end-to-end path accuracy, relative to
restriction-only hard trees (bare input, shared head) and flat readouts?

## Machinery (model-bound)

- Encoder: Qwen3.5-4B-UD-Q4_K_XL, llama-server 127.0.0.1:8998,
  `--embedding --pooling last`, 2560-dim. Utterances embedded verbatim
  (generated lowercase; no normalization layer).
- Embeddings on **id_slot 7** only — production readout slot 6 is never
  touched. Batch ≤ 32. Disk cache keyed sha1(text); cache is gitignored
  and must be deleted if the serving model/build changes.
- All artifacts regenerable from committed code + registered seeds;
  trained heads and caches are not committed (results JSONs are).

## Hierarchy

4 readout levels: 3 branches at L1 (smart_home / media / productivity),
×2 at L2, ×2 at L3, ×2 at L4 → 24 leaves; depth-d tree = prefix
(d=2 → 6 classes, d=3 → 12, d=4 → 24). Sibling leaves share object
phrases and verb families; ONLY the modifier phrase distinguishes them.

## Generator (deterministic, numpy default_rng, data seed 0)

utterance = "{verb} {obj} {modifier}". Verbs: train {set, turn, adjust},
held-out {make, put} (template transfer). Slots: rooms (6 train, 2
held-out), pct values, time values (train/held-out split). All EVAL
utterances use held-out verbs and/or held-out slots — every eval number
is template/slot transfer. Eval n=40/leaf.

**Markedness dial m** = P(modifier is the leaf-distinguishing one);
1−m draws from the L3 parent's generic pool (shared by both sibling
leaves, carrying no leaf information). At m=0 sibling leaves are
identically distributed → the deepest level's conditional accuracy has
a 50% chance floor by construction. m is a generation-time property:
each m config regenerates its own train/eval text.

**Pools are relabelings of one master set**: per leaf, pool_n train
utterances; an ancestor class at depth d = union of descendant-leaf
utterances relabeled. Embeddings are computed once per m config and
shared by every depth, arm, and sweep.

## Arms

- **A** — flat readout over the depth-d leaf classes (one trunk per
  depth), no hierarchy.
- **B** — hard tree, one shared trunk trained multi-task (CE summed
  along the gold path), bare 2560-d input, per-node centroids recorded
  from trunk outputs, cosine argmax. Restriction-only conditioning
  (wiresense design).
- **C** — identical to B but trunk input = concat[emb, one-hot(parent
  node)] (45-dim over all nodes; root level gets zeros, i.e.
  unconditioned exactly like B). Trained with gold parents
  (teacher forcing); evaluated BOTH with predicted parents (deployed
  number) and gold parents (oracle bound — the teacher-forcing gap).
- **C′** — FiLM variant: gamma/beta = Linear(one-hot) applied as
  emb′ = emb·(1+gamma)+beta before the unconditioned trunk; FiLM weights
  zero-initialized (starts as identity). Same gold/predicted protocol.
- **E** — B with beam width 2: prune to the top-2 (node, score) pairs
  per level, final argmax over frontier leaf scores. Registered
  exploratory (no adoption bar).

Trunk (all arms): Linear(D→2048) → GELU(tanh) → LayerNorm → Linear(→1024),
D = 2560 (B/A) or 2605 (C). Trained with per-level linear classifiers
(CE), which are then discarded; inference is centroid-cosine (mirrors
the wiresense arm-C contract). Adam lr 1e-3, batch 512, ≤40 epochs,
early stop patience 6 on val CE (10% of pool carved stratified, disjoint
from eval). Head-init seeds {0,1,2}; data seed fixed at 0.

## Sweeps

- **SWEEP-1 (primary, P2/P4)**: m=1.0, depth {2,3,4}, pool per class
  {16, 64, 256, 1024}, arms A(d-wise), B, C, C′, E. One trained trunk
  per (arm, pool) on the full depth-4 tree; depths evaluated by prefix.
- **SWEEP-2 (P3)**: depth 3, pool 256, m {0.5, 0.0}, arms B and C.
- **Robustness**: head seeds 1,2 at (m=1.0, d=4, pool=1024) for B and C.

## Metrics

L1 accuracy; per-level accuracy conditioned on correct parent; leaf path
accuracy (all levels correct); chance floors: L1 33.3%, deeper levels
50% (binary); at m=0 the leaf conditional floor is exactly 50%.

## Pre-registered predictions

- **P1 (replication guard)**: at m=1.0, pool ≥ 64, d=3: B path accuracy
  exceeds A flat accuracy by ≥ 3 points.
- **P2 (threshold)**: C−B gap at pool 16 is ≤ 3 points; the gap at pool
  1024 exceeds the gap at pool 16 by ≥ 3 points (conditioning is
  data-hungry, not uniformly helpful).
- **P3 (ceiling)**: at m=0.0 the deepest-level conditional accuracy is
  within ±5 points of 50% for BOTH B and C — conditioning shifts
  sample efficiency, never the information ceiling.
- **P4 (the hypothesis, primary)**: at pool 1024, m=1.0, predicted
  parents: (C−B path-acc gap at d=4) − (C−B gap at d=2) ≥ +5 points,
  i.e. the conditioning gap grows with depth. Verdicts:
  - PASS → conditioning is real and depth-protective; escalate (grow
    pools,FiLM-vs-concat choice, then real-data follow-up).
  - PARTIAL → gap grows only with gold parents (oracle), or by < 5
    points → conditioning real but eaten by teacher-forcing / noise;
    report the gap decomposition.
  - FAIL → gap ≤ 0 at pool 1024 → the hypothesis is dead in this
    regime; record and stop.

Amendments after data start require a dated section disclosing what and
why, before the affected results are committed.

## Amendment 1 (2026-10-07, before any sweep results were recorded)

The first launch was killed during pool-16 training when the embed log
revealed only 2,961 unique texts across all 25,536 generated — the
phrase/slot space was too small for "pool 1024" to mean 1024 distinct
examples, which would gut P2 (the pool threshold). Actions, in order:
the two sweep JSONs the aborted run had already written (pool 16 and 64)
were deleted UNEXAMINED; no sweep number was ever read. The generator
was widened (objs 2→6 per branch, leaf mods 3→8, generics 3→6, wider
slot values, 8 train / 2 held-out non-informative prefixes) and a guard
added: every leaf must have ≥1024 unique train texts (actual min 1152).
P1–P4, arms, sweeps, metrics, and decision rules are unchanged; only
the surface the dial acts on grew. Pool counts now mean
distinct-example capacity.

## RESULTS (2026-10-07, all six configs at commit 8b9c13a, seed 0 primary)

Runner summaries (path / leaf acc, predicted parents, oracle in parens):

- pool 16: d2 A=.902 B=.950 C=.926 C′=.893 E=.956 | d3 A=.886 B=.929
  C=.905 C′=.855 E=.940 | d4 A=.919 B=.784 C=.808 C′=.650 E=.795
- pool 64: d4 A=.942 B=.901 C=.909 C′=.812 E=.920 (d2/d3 similar shape)
- pool 256: d3 A=.991 B=.980 C=.972 C′=.953 E=.994 | d4 A=.952 B=.957
  C=.927 C′=.848 E=.970
- pool 1024: d2 A=.990 B=.982 C=.984 E=.991 | d3 A=.990 B=.982 C=.982
  E=.996 | d4 A=.959 B=.965 C=.972(orc=.972) C′=.831 E=.978
- m=.5, d3, pool 256: B=.932 C=.882(orc=.882) E=.950
- m=0.0, d3, pool 256: B=.923 C=.896(orc=.896) E=.952

Verdicts against the registered rules (adjudicate.py, mechanical):

- **P1: FAIL.** B never beats the flat head at d3 — A is ahead by
  0.7–1.5 points at every pool ≥ 64. In this generator a fully-marked
  leaf is so separable that restriction-only hierarchy adds nothing
  over flat supervision.
- **P2: FAIL.** The C−B gap SHRANK with pool: +2.4 pts at pool 16 →
  +0.7 at pool 1024 (registered direction was growth ≥ +3). The only
  positive conditioning signal is at the smallest pool, and it fades.
- **P3: FAIL as registered; substance confirmed post-hoc.** The
  registered metric (deepest-level conditional acc of the depth-3 tree)
  reads L3, which the dial does not mark (B=.981, C=.992 ≠ 50±5) — a
  pre-data design error, disclosed here. POST-HOC (deterministic rerun,
  posthoc_p3.py, labeled post-hoc) at the leaf level L4, m=0.0:
  B=.517, C=.517 — both on the 50% sibling floor; m=0.5: B=.705,
  C=.754, bracketing the ~75% information midpoint. Conditioning moves
  sample efficiency, never the ceiling — as claimed but at the level
  the dial actually acts on.
- **P4 (primary): PARTIAL by the letter — null in substance.**
  (C−B) gap grows d2→d4 by only +0.52 pts at pool 1024 (< the +5 bar;
  > 0 so not FAIL). Oracle == predicted exactly (no teacher-forcing
  gap anywhere — the walk almost never leaves the gold path), and the
  seed spread (B d4: .950–.965; C d4: .949–.972 over seeds 0–2) dwarfs
  the delta. At m=1.0 the depth-slope for B is just 1.8 pts: a
  fully-marked leaf leaves no depth-slope for conditioning to flatten.

Unregistered findings (measured, reported for completeness):

- **C′ FiLM is reliably harmful**: −13.4 pts vs B at d4/pool 1024,
  −10.9 at d4/pool 256, −8.9 at d4/pool 64. Zero-init did not save it.
  Do not FiLM frozen-encoder router embeddings.
- **Beam-2 (E) is the consistent small winner**: +1.4 pts over B at
  d4/1024, +1.8 at m=.5, +2.9 at m=0.0 — the cheapest lever, and it
  composes with the bare shared head (wiresense's registered beam-2
  recommendation replicates here).
- **Conditioning decomposes**: it helps the level it conditions on
  (+4.9 pts leaf-conditional at m=.5) but taxes the same shared trunk's
  other levels (d3 path C−B = −5.0 at m=.5), netting ≈ 0 end-to-end.
- **Small pools flip the hierarchy's sign**: at pool 16 the tree beats
  flat at d3 (B .929 vs A .886, the registered-P1 direction) but
  collapses at d4 (B .784 vs A .919) — with scarce data the walk
  amplifies L1 errors multiplicatively (B's L1 at pool 16 is only
  .953, so ~14% of evals die before L4) while flat supervision pools
  everything into one readout. Seed noise at pool 16 is unmeasured;
  report as observed.

## Conclusion

The multi-level decision tree is bounded by utterance information
(markedness), not by readout conditioning: when the leaf is marked,
even a flat head scales to 24 classes at depth 4 (95.9%); when it is
not, every arm sits on the sibling floor (≈50%) regardless of
conditioning. Parent-one-hot conditioning neither rescues depth nor
hurts the ceiling; its one positive blip is low-data (+2.4 pts at pool
16, fading with pool). The scalable levers measured here are beam-2
and markedness itself — consistent with the wiresense diagnosis that
the think-level collapse is an information problem, not head capacity.
Per the registered rule, P4 is not PASS: no escalation of concat
conditioning; recorded and stopped.

## Amendment 2 (2026-10-07, registered before any deep-tree data)

Question: what is the maximum depth that still holds acceptable
end-to-end accuracy? The 4-level tree could not answer it, so the
generator extends mechanically: every depth-4 leaf splits on two
further binary axes appended as trailing modifier tokens — L5
"with full detail" / "in brief", L6 "urgently" / "eventually".
Axis words are lexicalized and shared across all branches (the
question each level asks is literally in the text — best-case
marking, same spirit as the pct markers at L4), are train-only (no
held-out variant), and the m-dial's fallback beyond L4 is the shared
generic "as usual". Result: 96 leaves, 190 nodes, 80,067 unique
training texts.

Registered config: m=1.0 only; pool 256 per leaf-class (per-class
parity with the depth-4 evidence, where pool 256 was within 0.8 pts
of pool 1024); arms: B (one shared trunk, per-level heads at levels
1–6, hard walk), A flats at d4 and d6, E beam-2 on B; B head seeds
1,2 for spread. Metrics: path accuracy at d = 2..6 and per-level
conditional accuracies; "maximum acceptable depth" is read off the
curve at the 95% and 90% bars. No hypothesis bar is registered —
this is a measurement sweep.

Disclosures: (1) the conditioning one-hot basis is now the deep
registry (189 nodes); C/C′ are not run (dead per RESULTS), and B/A/E
are unaffected by the basis. (2) Depth-2..4 numbers inside this run
are internally comparable but not byte-identical to SWEEP-1 (the
extra trailing tokens are distractor surface for shallow readouts);
shallow defaults are preserved so the frozen SWEEP-1/2 results remain
reproducible from the committed code.

## RESULTS 2 — depth extension (2026-10-07, commit 9e792c2)

Walk accuracy by depth (m=1.0, pool 256/leaf, seed 0; classes are
3/6/12/24/48/96; A = flat at that depth, B = hard walk, E = beam-2):

- d2: B .903 E .914
- d3: B .894 E .921
- d4: A .813 B .826 E .856
- d5: B .825 E .859
- d6: A .899 B .825 E .859

B per-level conditional accuracies in the d6 walk: L1 .918, L2 .984,
L3 .990, L4 .924, L5 .9997, L6 1.0000. Product .825 ✓ matches the
measured path accuracy. B seed spread at d6: .825 / .831 / .844.

Findings:

- **Depth stopped costing anything after level 4.** The two new
  binary levels (48 → 96 leaves) were free: per-level conditional
  accuracy .9997 and 1.0000 — lexicalized questions are trivially
  readable, exactly as designed for best-case marking.
- **The cost concentrates in the COARSEST and the sibling-pair
  levels**, not with depth: L1 (3-way) .918 and L4 (24-way sibling
  discrimination) .924 account for the entire ~17-point loss.
- **The walk, not depth, became the wall.** Compounding L1's 91.8%
  caps the walk at .825 no matter how perfect the deeper levels are,
  and at 96 classes the flat readout overtakes it (A_d6 .899 vs
  B .825; at 24 classes they were tied, .813 vs .826). Beam-2
  recovers +3.4 pts (.859) by keeping two first-level candidates.
- Distractor note (disclosed above): the trailing axis tokens also
  depress shallow readouts in this run (d2 B .903 vs .980 in
  SWEEP-1), so cross-run comparisons are invalid; within-run
  comparisons are the evidence.

Answer to the registered question, in the best-case marking regime:
there is **no depth wall through 6 levels / 96 leaves** — marginal
depth cost after level 4 is ~0. The wall that appears at 96 classes
is structural: sequential walks lose to direct 96-way readouts once
the first-level error rate (8%) exceeds what the deeper levels can
afford to multiply. Deployment shape implied: use the walk for
interpretability with beam-2, or go flat when the class count gets
large and the coarsest level is the weak link; in both cases the
binding constraint remains per-level information, not depth.

## Transfer note (2026-10-07, post-study)

The beam-2 recommendation was deployed in wiresense's production
router (2-level walk, frozen evals) — `tools/eval_router_beam.py`
there. Two findings transfer-relevant to this lab:

- The gain replicated (root 57/62 vs greedy 56/62, live-path
  composite 13/26 vs 12/26), but ONLY with length-normalized mean
  path scores. This lab's walks were equal-depth everywhere, so the
  cumulative-sum formulation never met the pitfall: with paths of
  different depth, raw sums let the deeper path structurally outbid
  the shallower one (wiresense measured 43/62 vs greedy 56/62 — a
  13-point crash from scoring, not from the mechanism).
- The rescue pattern matched lab E: the win case was a coarse-level
  runner-up whose deeper evidence was strong ("Choose a name for my
  goldfish": root ranks think #2, the think.tool child wins on the
  mean) — the same "beam keeps two first-level candidates" effect
  seen at L1 here.
