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
