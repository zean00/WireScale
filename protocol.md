# WireScale: does parent-conditioning make multi-level decision trees scale?

Registered 2026-10-07, before any data. Standalone testbed (deliberately
beyond WireSense): synthetic utterance-marked hierarchy, frozen real
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
  (WireSense design).
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
the WireSense arm-C contract). Adam lr 1e-3, batch 512, ≤40 epochs,
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
  composes with the bare shared head (WireSense's registered beam-2
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
and markedness itself — consistent with the WireSense diagnosis that
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

The beam-2 recommendation was deployed in WireSense's production
router (2-level walk, frozen evals) — `tools/eval_router_beam.py`
there. Two findings transfer-relevant to this lab:

- The gain replicated (root 57/62 vs greedy 56/62, live-path
  composite 13/26 vs 12/26), but ONLY with length-normalized mean
  path scores. This lab's walks were equal-depth everywhere, so the
  cumulative-sum formulation never met the pitfall: with paths of
  different depth, raw sums let the deeper path structurally outbid
  the shallower one (WireSense measured 43/62 vs greedy 56/62 — a
  13-point crash from scoring, not from the mechanism).
- The rescue pattern matched lab E: the win case was a coarse-level
  runner-up whose deeper evidence was strong ("Choose a name for my
  goldfish": root ranks think #2, the think.tool child wins on the
  mean) — the same "beam keeps two first-level candidates" effect
  seen at L1 here.

Scope note (2026-10-07, user decision): extending this routing into a
multi-agent router (spawning specialized sub-agents per intent, of the
kind orchestrator LLMs do by generation) stays LAB-ONLY CURIOSITY —
WireSense deliberately stays simple. The measured production transfer
of the beam-2 mechanism is tool selection instead (WireSense
PROTOCOL-mcp-selection.md MS-20).

## S-series (registered pre-data): sub-agent routing

Question (2026-10-07, user-directed): modern agent frameworks spawn
many specialized sub-agents; could a decision tree route user requests
to them — and where do beam-2 and flat readouts sit in that shape?
Scope: LAB-ONLY curiosity (see scope note above); the substrate is the
production mechanism (raw slot-7 embeddings + centroid cosine, no
trained heads), which is what WireSense's router and MCP gate actually
run.

Shape: 8 domains × 8 agents = 64 sub-agents. Each agent owns unique
specialty phrase pairs; each domain owns shared flavor vocabulary;
utterances are prefix + phrase + qualifier. Specialization dial m =
P(utterance carries the agent's specialty pair); 1−m draws a
domain-flavored generic request (domain visible, agent NOT nameable).
Refs: 32/agent (same m-distribution as eval) + 32 dedicated
domain-generic refs/domain (domain readout stays strong at every m —
production-faithful: coarse nodes have their own pools). Eval: 64
held-out per agent per m. Seeds fixed; per-agent uniqueness ≥ 448.

Arms (all centroid-cosine, L2 norm):
- A_flat — argmax over all 64 agent centroids
- B_walk — greedy two-level: domain top-1 → agent top-1 inside it
- B_oracle — gold domain → agent top-1 (two-level's ceiling)
- E_beam2 — domain top-2 → agent top-1 per path, mean(cos_dom, cos_agent)
  picks (mean, never sum — the router-walk length-bias lesson)

m ∈ {1.0, 0.5, 0.0}. No adoption bar (lab): this is mechanism
cartography for the parked sub-agent idea.

Registered hypotheses:
- S1 (beam replication at agent scale): E_beam2 − B_walk ≥ +1 pt at
  m=1.0 AND m=0.5.
- S2 (coarse-first vs flat at 64 agents): OPEN — measured, no bar. The
  depth-6 wall said a 6-level walk loses to flat at 96 classes; a
  2-level walk compounds ONE coarse decision, so the shape may survive
  agent counts. Either verdict is informative.
- S3 (specialization floor): at m=0, agent accuracy ≈ 1/A (12.5%) for
  EVERY arm including B_oracle, while domain accuracy stays ≥ 90% —
  requests that never name the specialty are unroutable at the agent
  level regardless of structure (the m=0 sibling floor, one level up).

Runner: subagents.py (self-contained; embed_client slot 7, cached).
This section + runner committed before any scored run.

Amendment 1 (2026-10-08, pre-data — no valid scored run exists):
pass 1 (archived as results/subagents_pass1_invalid.json) is
discarded for two independent reasons, both fixed before the rerun.
(a) Scoring bug: the three walk arms compared the pick's global index
against the gold agent NAME — always false, so B_walk/B_oracle/E_beam2
recorded exactly 0.0 (impossible as a measurement; A_flat and
domain_acc are unaffected and were real). (b) Substrate flaw vs this
section's own premise ("domain readout stays strong at every m"):
domain pools were built from generic dom-pair texts only, so the
domain level could not see specialty vocabulary — domain_acc 0.3879 at
m=1.0 (0.5217/0.6655 at m=0.5/0.0), domain-centroid cross-cos 0.9582.
Fix: domain pools now draw from the domain's FULL vocabulary (dom
pairs + every member agent's specialty pairs), mirroring production
coarse nodes whose refs include subclass utterances. Bars and
hypotheses unchanged. The A_flat/domain numbers of pass 1 stand as the
record of the rejected substrate.

Amendment 2 (2026-10-08, still pre-data): the template itself was the
remaining flaw. Pass-1's shared filler (prefix + qualifier identical
across ALL classes) swamped the signal: domain-centroid cross-cos
0.958, A_flat 0.35 at m=1.0 despite globally unique specialty pairs.
Probe series on frozen cache (no scored run): pure pair texts classify
domain at 1.000 (lexicons inherently separable); prefix + TWO pairs
lifts m=1.0 to domain 0.820 / A_flat 0.906 / B_oracle 0.926 (single
pair: 0.43/0.35/0.51). Template therefore becomes prefix + pair +
pair; the qualifier slot is dropped. Composition probes that did NOT
help are recorded as rejected: dom vocab 4->8 pairs (m=0 generic
regime 0.844 -> 0.781), dom-weighted pools 50/75% (0.844/0.805, and
m=1.0 domain drops to 0.799/0.648) — m=0 sits at 0.78-0.84 under every
composition, a held-out generalization ceiling of short synthetic
generic texts, not a composition artifact. dom widens only 4->6 pairs
(the minimum that keeps the m=0 draw universe above the pool: 6
prefixes x 36 two-pair combos = 216 >= 2x96 draws; at 4 pairs the
universe EQUALS the pool and the dedup loop starves). The registered
"uniqueness >= 448" clause is re-derived on its original purpose for
the new template: per-agent universe 6 x 6^2 = 216 >= 192. Arms,
sizes, seeds, hypotheses, and bars are all unchanged. Honest
prediction on record: S3's domain >= 0.90 half will likely FAIL at
~0.84 (ceiling above); its agent-floor half (every arm ~ 12.5%) is the
primary claim.

### S-series results (2026-10-08, scored run; runner d571ce8, data results/subagents.json)

64 agents x 4096 evals per m, slot-7 raw embeddings, centroid cosine.

| m | A_flat | B_walk | B_oracle | E_beam2 | domain_acc |
|------|--------|--------|----------|---------|------------|
| 1.0 | 0.9287 | 0.7961 | 0.9534 | 0.8730 | 0.8364 |
| 0.5 | 0.4800 | 0.4172 | 0.5017 | 0.4497 | 0.8440 |
| 0.0 | 0.1169 | 0.1021 | 0.1182 | 0.1116 | 0.8555 |

Verdicts (mechanical, per registered bars):
- S1 @m10: beam2-greedy = +0.0769 (PASS bar +0.01)
- S1 @m05: beam2-greedy = +0.0325 (PASS bar +0.01)
- S3 @m00: arms [0.1169, 0.1021, 0.1182] vs 1/A = 0.125 (PASS — every
  arm lands at the floor, oracle included); domain_acc = 0.8555
  (FAIL bar 0.90 — as predicted in Amendment 2, the generalization
  ceiling of short synthetic generic texts; measured slightly above
  the probed ~0.84)

S2 (open, flat vs coarse-first at 64 agents): A_flat beats B_walk at
EVERY m (0.929 vs 0.796 fully nameable; 0.480 vs 0.417 at m=0.5; both
at floor at m=0). The depth-6 wall replicates one level up: the walk
pays its coarse decision's full error rate, B_walk ≈ domain_acc x
agent_acc|domain (m=1.0: 0.836 x 0.952 = 0.796, exact), and no amount
of agent-level skill recovers a lost coarse decision. E_beam2 claws
back roughly half the coarse-level gap (+0.077 of 0.157 at m=1.0) but
never overtakes flat.

Mechanism reading for the parked sub-agent idea:
- Nameability is the whole game: at m=1.0 flat routing hits 0.93 at
  64-way (chance 0.016); at m=0 NOTHING routes (every arm ~ 0.11-0.12
  ≈ 1/8) including the oracle — a router cannot pick an agent the
  utterance never names. Any real sub-agent router needs utterances
  that carry agent-identifying content, or it must fall back to
  domain-level service.
- Coarse-first is a liability, not an engine, at this scale: it caps
  accuracy at the coarse level's error rate and compounds errors
  downstream. Beam-2 measurably repairs a two-level walk (S1, 3-for-3
  with the WireSense router and MS-20), but "flat when embeddings are
  strong, walk only when the coarse level is near-ceiling" is the
  measured shape. Production note: WireSense's root level runs 0.919
  (57/62) on real utterances — above this substrate's 0.84-0.86
  synthetic ceiling — so the flat-vs-walk gap narrows with a better
  coarse level but the direction held at every measured m.
- The substrate verdict is part of the result: pass 1 (archived) shows
  shared template filler dominates raw slot-7 embeddings of short
  synthetic texts (centroid cross-cos 0.958, domain_acc 0.39); content-token
  mass, not vocabulary size or pool weighting, is what makes centroid
  readouts work — consistent with production refs being full natural
  utterances.

## R-series (registered 2026-10-13, pre-data): chunk-relevance readouts

Question (user-directed): a document is split into ~20 chunks; for each
(question, chunk) pair a decision layer judges "does this chunk carry
information relevant to — or potentially answering — the question";
yes-chunks are collected as answer-source candidates. Does pointwise joint
judgment beat bi-embedding cosine (the vector-index mechanism, exact at
this scale), especially when the question does NOT share the chunk's
vocabulary? Scope: WITHIN-document chunk selection on this repo's own
committed prose. Cross-document routing is the S-series' measured shape;
corpus-scale ANN is out of scope (exhaustive flat won every small-N
comparison in P/S — nothing to prune).

### Corpus (R-1)

This repo's markdown at the registration commit, verbatim, no
normalization: README.md, protocol.md, PROTOCOL-mcp-selection.md,
models/README.md (~140 chunks expected). Chunking deterministic: ~1100
chars, 150 overlap, boundaries snapped to whitespace. Chunk counts are
recorded at freeze; a doc shorter than 2 chunks is dropped.

### Questions (R-2)

LLM-generated on the same lab server (slot 7, temp 0.8, per-call seeds
derived from a registered base), two recipes per chunk, reusing the MS
eval split as the markedness dial:

- **direct** — one question (≤20 words) whose answer the passage states,
  reusing the passage's own vocabulary (query names the target).
- **indirect** — one question (≤20 words) describing the underlying need
  or goal ONLY; the passage's vocabulary is not reused (the m=0 direction:
  zero surface overlap, information present in the pair).

12 train + 12 eval chunks per doc (seeded, disjoint; small docs split what
exists, min 2 each where chunks permit — a 2-chunk doc splits 1+1, which
is what the frozen run did for models/README.md). Plus 8 hand-written off-corpus negatives (math,
chit-chat, and two production-owned domains — timer/weather) judged
against README.md's chunks (the small-doc regime; abstention is the point,
not scale). Question files are generated by `chunkrel.py --gen-questions`,
HAND-AUDITED, then frozen: no edits after any scored run (MS-4 rule).
Generation output is registration infrastructure, never gate data.

Audit (2026-10-13, pre-freeze, pre-data — no scored run exists yet):
all 116 candidates read against their gold chunks. 9 INDIRECT eval
questions rewritten — generator filler with no mapping to the gold
chunk (indices, old→new text, rationale recorded verbatim in
`results/r_questions.json` → `audit`): eval #11, #15, #19, #25, #29,
#31, #33, #39, #53. Rewrites follow the registered indirect recipe
(need only, ≤20 words, no passage vocabulary). Direct eval questions:
no corrections — several awkward phrasings are verbatim-anchored in
their gold chunks and left as generated. Train set kept as generated:
a similar filler rate and a few plausible wrong-chunk assignments
remain, which only weaken J-combo training and τ tuning (conservative
for R2); no adjudicated metric reads train. The question file is
frozen at the commit carrying this note.

Incident + Amendment (2026-10-13, disclosed before any valid data).
Run 1 (19:15–19:33, exit 0) is **INVALID**: the audit note above was
added to `protocol.md` — a corpus document — after question
generation, shifting its chunking 32 → 33 chunks. Measured drift: 8
eval gold chunks and 2 train golds (protocol.md indices ≥ 26) pointed
at shifted text, so those questions were judged against the wrong
passages. Root cause is operator edit-ordering, not the runner; the
drift was caught during post-run adjudication, before any result was
committed or recorded. Run 1's output is kept as
`results/chunkrel_run1_invalid.json` and is never scored. Amendment:
`chunkrel.py` now pins the corpus to commit `5146e2d` — the exact
chunking the freeze audit verified — via `CORPUS_COMMIT` /
`_pinned_corpus` (working-file divergence logs and uses the pinned
blob). Questions remain frozen and untouched. Run 2 against the
pinned corpus is the scored record.

### Arms (R-3)

All arms see the question and the chunks of its OWN document only.

- **V** — bi-embed (slot 7, cached, fp16→fp32, L2), cosine. The
  vector-index stand-in, exact, no index to build. presence@1/@3;
  abstention via threshold τ tuned on TRAIN pairs (best pair-classification
  accuracy), frozen before eval.
- **J-llm** — planner reads question + full chunk text in one context and
  answers yes/no (null-framed per MS-26), temp 0, ≤16 tokens, thinking
  disabled (chat_template_kwargs; verified on the lab server). Exhaustive
  over the doc's chunks. p_yes = normalized first-token yes/no logprob
  mass when the server serves logprobs (else binary; coverage reported).
- **J-combo** — frozen pair head, the arm that scales like a vector index
  (chunk embeddings cacheable at ingest): features [q; c; |q−c|; q·c]
  (L2'd, 10240-d) → Linear 2048 → GELU(tanh) → LayerNorm → Linear 512 →
  logit. Trained on train pairs: gold=1; negatives=0 (2 same-doc non-gold
  + 1 wrong-doc per question). Adam 1e-3, batch 64, ≤40 epochs, patience 6
  on val CE (10% carve), seed 0. presence@1/@3 by sigmoid rank; abstain
  at 0.5 fixed.
- **J-menu** — listwise control: ONE call, numbered previews (first 200
  chars) of a seeded 20-chunk subset of the doc CONTAINING the gold chunk
  (measures picking, not pruning — same contract as MS manifests); reply
  is the index, or 0 if none (abstain). Tests pointwise isolation vs
  attention dilution at N=20.

### Metrics (R-4)

presence@1/@3 (ranked arms; J-llm reports gold-in-yes-set recall and
mean yes-set size — its within-set ranking is only the best-effort p_yes);
per-recipe breakdown (direct/indirect — the P3 lesson: record the level
the dial acts on); abstain-correct and false-collect on the 8 hand
negatives; mean prompt+completion tokens and wall-clock ms per question
(J-llm sums its exhaustive per-chunk calls); ECE (10 equal-width bins)
for the probabilistic scores (J-combo sigmoid, J-llm p_yes) over all
evaluated pairs. V has no probability and is excluded from ECE (cosine
miscalibration is already adjudicated — production tree-probe).

### Registered predictions

- **R1 (primary — pairing repairs markedness)**: on INDIRECT questions,
  J-llm gold-in-set − V presence@3 ≥ +10 pts; on DIRECT, within ±3 pts;
  guardrail: J-llm abstain-correct ≥ 4/8 on hand negatives (a
  yes-everything judge is not a win). Verdicts: ≥+10 PASS; +3..+10
  PARTIAL (direction real, magnitude small); ≤+3 or direct clause loses
  >3 pts FAIL → pairing does not repair low-markedness retrieval on this
  stack; record and stop (P4 precedent).
- **R2 (economics)**: J-combo presence@3 keeps ≥ 50% of the (J-llm − V)
  indirect gap. PASS → the cacheable head is the deployable readout;
  FAIL with R1 PASS → joint LLM read required at low markedness, J-combo
  demoted to pre-filter.
- **R3 (isolation vs dilution)**: J-menu presence@1 does not exceed
  J-llm gold-in-set recall by more than +2 pts (pointwise isolation ≥
  listwise dilution at N=20). Either direction informative.
- **R4 (calibration)**: ECE ≥ 0.05 for at least one of {J-combo,
  J-llm} → miscalibration replicates; collect by rank, not threshold.

### Disclosures

1. Question generation and J-llm judgment use the same 4B model on slot
   7 — generator and judge are one family; a stronger judge could move
   R1's magnitude (not registered, stated bias).
2. Self-corpus: questions are about this repo's own experiments; the
   generator's domain familiarity is a stated bias (real-prose substrate
   chosen deliberately — the S-series synthetic ceiling 0.78–0.84 vs
   production 0.92).
3. No end-to-end answer-quality metric — needs an answer rubric; out of
   scope. Presence is the gate metric (MS presence precedent).
4. J-llm cost is dominated by PROTOCOL-mcp-selection.md (~24 eval
   questions × ~110 chunks ≈ 2.6k judgments); expected tens of minutes,
   sequential, temp 0.
5. J-combo seed spread unmeasured at v1 (seed 0 only); report as
   observed, as with pool-16.
6. τ (V abstain) and all J-combo training use TRAIN split only; eval
   questions never train anything.

Runner: `chunkrel.py` (self-contained; `--smoke` marked and never
scored; `--gen-questions` writes the freeze candidates; default requires
the frozen files). This section + runner committed before any scored run.

## R-series RESULTS (2026-10-13, scored — run 2, pinned corpus)

Run: `chunkrel.py` default at commit f183496 (corpus pinned to 5146e2d
per the incident above; run 1 INVALID, kept as record). Questions
frozen at 657d89b. p_yes coverage 1.0 (first-token logprobs served on
every judgment). Wall ~17.5 min. File: results/chunkrel.json.

| arm | presence@1 | presence@3 | direct | indirect | abstain (8 negs) | prompt tokens | ms/q |
|---|---|---|---|---|---|---|---|
| V bi-embed cosine | .069 | .172 | .207 | .138 | 0/8 | — (cached) | — |
| J-combo pair head | .000 | .069 | .069 | .069 | 8/8 | — | — |
| J-llm exhaustive judge | — | .517 (recall) | .793 | .241 | 8/8 | 1.70 M | 17100 |
| J-menu pick-from-20 | .328 | — | .345 | .310 | 8/8 | 71 k | 555 |

Mean yes-set size: 3.4 (direct) / 5.9 (indirect); on the 137-chunk
doc 9.6, max 42. τ(V) = 0.204 (train pair acc .556 — the tune itself
flagged the comparator); J-combo valBCE .641 ≈ ln 2.

POST-HOC context: random within-doc baselines for this eval mix
(8 README / 24 protocol / 24 MS / 2 models questions over 9/32/137/2
chunks) are @1 ≈ .048 and @3 ≈ .128. V sits at 1.3–1.4× random at
both recipes; J-combo at ~0.5× random.

Verdicts (mechanical, as coded and committed pre-data):

- **R1: PARTIAL.** Indirect gap J-llm − V@3 = **+10.3 pts** (clears
  the +10 bar by 0.3); guardrail 8/8 (≥ 4/8); the direct-parity
  clause (|J-llm − V@3| ≤ 3 pts) broke at **+58.6 pts — in J-llm's
  favor**. Design note (P3 precedent, disclosed): the parity clause
  presumed V competent at high markedness; measured V is random-level
  at BOTH recipes, so "within ±3 of V" had no competence to anchor
  to. The registered PARTIAL gloss ("direction real, magnitude
  small") does not describe this PARTIAL — the gap cleared the bar;
  the conjunct failed. Substance: J-llm strictly dominates V on
  every recipe, doc, and metric.
- **R2: FAIL.** J-combo keeps **−67%** of the indirect gap (p3 .069
  vs V's .138 — worse than the comparator it was meant to cheaply
  replace). Registered reading applies in substance: the cacheable
  pair head is NOT the deployable readout at this train size (230
  pairs, valBCE ≈ chance); joint LLM read required; J-combo demoted
  to pre-filter at best. Its one clean behavior: abstains 8/8
  off-corpus.
- **R3: PASS** (menu does not beat exhaustive): .328 ≤ .517 + .02.
  Isolation loses to exhaustive reading. Post-hoc nuance: menu
  scores 0.0 on the two small docs (nothing to pick from at 9 and 2
  chunks) and .375/.417 on the big ones — at 24× fewer tokens.
- **R4: PASS** (miscalibration real): ECE .098 (J-llm), .137
  (J-combo) — both ≥ .05; neither arm's probabilities are usable as
  confidence without recalibration.

Readings (post-hoc, labeled):

- The registered frame "pairing repairs markedness" is confounded by
  comparator collapse: slot-7 bi-embedding cosine ranks question→chunk
  at random within-doc on this corpus at BOTH markedness levels —
  consistent with the S-series substrate verdict (template filler
  dominates raw slot-7 embeddings) and the adjudicated cosine
  miscalibration. What R1 measured is dominance, not repair.
- Markedness still taxes the judge itself: J-llm recall .793 → .241
  direct → indirect. The information story survives — inside the
  joint reader, not between architectures.
- Per-doc inversion (the deployment cut): J-llm does best exactly
  where V collapses (MS doc .625 vs .000; protocol .458 vs .208) and
  worst where V is tolerable (README .25 vs .375). Exhaustive joint
  reading buys its 17 s/q precisely at scale.
- Over-collection is the judge's failure mode: mean yes-set 9.6 of
  137 chunks on the MS doc (max 42) — precision, not recall, is the
  next target if this line continues.
- Audit cross-check: the 9 pre-data-rewritten indirect questions went
  3/9 for J-llm vs 4/20 for the untouched indirects — the rewrites
  did not inflate the arm they might have favored.

Run 1 (invalid) vs run 2: identical verdict letters; the drifted run's
indirect gap read +13.8 (drift inflated it) — +10.3 is the number of
record.

Close-out: no registered adoption bar was met (R1 PARTIAL, R2 FAIL);
nothing is adopted and nothing further is registered in this series.
The narrow measured statement: exhaustive joint judgment is the only
readout on this stack that retrieves above chance — at exhaustive
cost (1.70 M prompt tokens over 58 questions), with miscalibrated
confidence and scale-driven over-collection.

## V-series (registered 2026-10-13, pre-data): the comparator autopsy

R-series measured V (bi-embed cosine, slot 7, last-pool) at
random-chance within-doc ranking (p3 .172 vs random ≈ .128),
confounding R1's frame. Before any markedness claim is re-asked:
WHY is the vector readout dead — pooling, input format, or length
asymmetry?

Questions: the R-series frozen eval set (58), unchanged.

Substrate note (disclosed): same model file and quant as the frozen
encoder. The MEAN-pooled arms are served by a second llama-server
instance (127.0.0.1:8999, `--pooling mean`, CPU-only — the GPU is
fully occupied by the production server, which is untouched). This is
a readout change, not a model change. Mean-pooled vectors use a
SEPARATE cache file and are never written to the shared slot-7 cache.

Arms (all bi-embed + cosine; presence@1/@3 by recipe; no τ/abstention
— ranking only):
- **V0** — last-pool, no prefix (R-series V replicated; sanity: p3
  within ±.03 of R run 2's .172, else flag).
- **V1** — last-pool + prefixes: question embedded as `query: {q}`,
  chunk as `passage: {chunk}` (chosen a priori, generic).
- **V2** — mean-pool, no prefix.
- **V3** — mean-pool + prefixes.
- **V4** — last-pool, length fix: question vs the chunk's first
  300 chars (word-snapped).

Predictions (mechanical):
- **VA (primary)**: best non-V0 arm's indirect p3 − V0 indirect p3
  ≥ +10 pts PASS (comparator rescuable → R1 re-askable);
  +3..+10 PARTIAL; < +3 FAIL → bi-embedding retrieval dead on this
  substrate at any pooling/format; R1 stands as dominance-only.
- **VB (driver)**: mean-effect = mean(V2,V3) − mean(V0,V1) on
  indirect p3; prefix-effect = mean(V1,V3) − mean(V0,V2). The larger
  is the named driver; both < +3 pts → neither dial moves it.
- **VC (length)**: V4 direct p3 − V0 direct p3 ≥ +5 pts PASS, else
  FAIL.

Disclosures: prefixes are generic, not tuned to this model family;
the 300-char cut is arbitrary but registered; no training anywhere in
V-series; mean arms depend on a second server (if it cannot run, the
amendment drops V2/V3 with disclosure, VA/VB adjudicated on the
remaining arms).

## W-series (registered 2026-10-13, pre-data): retrieval walk vs flat

The founding question (P/S verdict: information beats structure when
reads are equally cheap) re-asked in the one regime where the read is
EXPENSIVE: exhaustive J-llm judging costs ~29k prompt tokens / ~17 s
per question. Does a two-level walk (cheap window pick → judge
inside) buy acceptable recall at a fraction of the cost?

Corpus/questions: the R-series pinned corpus (5146e2d) and frozen 58,
unchanged. Windows: consecutive non-overlapping groups of 20 chunks
in doc order (README 9 → 1; protocol 32 → 2 [20+12]; MS 137 → 7
[6×20+17]; models 2 → 1). Where a doc has one window the walk
degenerates to flat (recorded per doc).

Arms:
- **FLAT** — exhaustive J-llm over all chunks (fresh same-session
  rerun; cross-run sanity: per-question recall agreement with R run 2
  ≥ 75%, else flag before adjudication).
- **WALK** — level 1: menu call over window previews (label "chunks
  a–b" + first 200 chars of the window's first chunk; pick 1..m or
  0); level 2: exhaustive J-llm inside the picked window. k=0 counts
  as recall 0 with abstention recorded.

Metrics: recall direct/indirect/all; P(gold's window picked) (first
step); P(gold in yes-set | window picked) (conditional); prompt
tokens; ms/q. Post-hoc descriptive from persisted per-pair p_yes:
yes-set size vs threshold sweep (t ∈ .5–.9). Per-pair scores are
PERSISTED (R's runner discarded them; this one stores them — new
runner, frozen chunkrel.py untouched).

Predictions (mechanical):
- **W1 (accuracy)**: FLAT − WALK ≥ 0 on all-recall → PASS (flat does
  not lose); |diff| ≤ 2 pts PARTIAL; WALK > FLAT by > 2 pts FAIL
  (structure wins accuracy too — would be a first on this stack).
- **W2 (economics)**: WALK recall ≥ 0.8 × FLAT recall AND WALK
  tokens ≤ 0.35 × FLAT tokens → PASS (structure's first economics
  win); else FAIL (record: structure loses even when reads are
  expensive).
- **W3 (mechanism)**: P(yes | window picked) ≥ 0.9 × FLAT recall →
  PASS (the walk is capped by its first step, not diluted inside —
  the P-series shape); else FAIL.

Disclosures: window previews are deliberately crude (first 200 chars
of the window's first chunk) — a weak walk is part of the test; FLAT
rerun burns ~1.7 M tokens for same-session comparability; judge temp
0 is not bit-deterministic across sessions (MS softhint precedent).

## J-series (registered 2026-10-13, pre-data): pair-head scaling

R2's FAIL came at 230 train pairs (58 questions). Sample size or
mechanism? Generate 8 questions per (train chunk, recipe) with the
R-series generator machinery (temp .8, seeded — new seed base, slot
7) → 29 train chunks × 2 recipes × 8 = 464 train questions → ~1.9k
pairs (R-series recipe: gold=1; 2 same-doc non-gold + 1 wrong-doc =
0). Retrain J-combo (identical architecture/optimizer, seed 0).
Evaluate on the SAME frozen 58 eval (never trained on).

Train questions are generator output, UNAUDITED (R-series train
policy: train-only, no adjudicated metric reads train; similar filler
rate — disclosed).

Arms: **V0** (cosine, replicated), **J1** (1× train — the R-series
pairing, retrained same-session for comparability), **J8** (8× train).
Per-chunk scores persisted per question.

Predictions (mechanical):
- **JA (primary)**: J8 p3 (all) − V0 p3 (all) ≥ +5 pts PASS (the
  head learns the task with data; R2 was sample size); ≥ 0 PARTIAL;
  < 0 FAIL (mechanism dead at any tested scale).
- **JB (markedness generalization)**: J8 indirect p3 ≥ 0.5 × J8
  direct p3 PASS, else FAIL.

Cost: ~464 generations (~20 min sequential), embeddings cached,
training ~1–2 min, no judge calls. Disclosures: 8× is one point on
the curve, not a sweep; architecture untouched (capacity binding
would conflate — noted, not tested); eval frozen → no selection on
eval.

## V-series RESULTS (2026-10-14, scored — commit 19fe2a3, pinned corpus)

Run: vautopsy.py, single scored pass, no reruns. Sanity: V0 all p3
.1724 = R run 2 exactly (in tolerance). Mean arms served by the
8999 CPU instance (verified: same text, cos(last-pool, mean-pool)
= .61 → the poolings genuinely differ).

Arms (p3 by recipe; p1 all in parens):

| arm | direct | indirect | all (p1)  |
|-----|--------|----------|-----------|
| V0  | .207   | .138     | .172 (.069) |
| V1  | .172   | .103     | .138 (.086) |
| V2  | .207   | .172     | .190 (.155) |
| V3  | .207   | .138     | .172 (.103) |
| V4  | .138   | .138     | .138 (.103) |

Verdicts (mechanical):
- **VA: PARTIAL** — best non-V0 arm on indirect p3 is V2 (.172);
  gap +3.45 pts → the +3..+10 band. The comparator moves, but not
  enough to re-ask R1.
- **VB: driver = pooling** — mean-effect +3.45, prefix-effect
  −3.45. MAGNITUDE TIE (disclosed): pooling gains exactly what
  prefixes lose; the sign is the only difference. Registered rule
  names the larger; the positive dial is pooling.
- **VC: FAIL** — V4 direct .138 vs V0 .207 (−6.9 pts). Cutting the
  chunk to its first 300 chars destroys direct recall.

Post-hoc readings (not verdicts): V2's gain is mostly top-1 (p1
.069 → .155, ×2.2); @3 barely moves. Even the best embedding
configuration (.190 all p3) sits barely above the @3 random
baseline (.128) and at ~37% of the flat judge (.517). The autopsy
answer: no pooling/format/length dial rescues bi-embedding on this
substrate — the encoder's question-vs-passage geometry is the
ceiling, confirming R1 as dominance-only.

## W-series RESULTS (2026-10-14, scored — commit 19fe2a3, pinned corpus)

Run: walkret.py, single scored pass. Sanity: FLAT per-question
recall agreement with R run 2 = 1.0 (bar: ≥ .75) — the judge is
stable across sessions at temp 0 on this corpus.

| arm    | direct | indirect | all   |
|--------|--------|----------|-------|
| FLAT   | .793   | .241     | .517  |
| WALK   | .138   | .034     | .086  |

Economics: WALK tokens 1.7k vs FLAT 29.2k per question (×.058);
ms 1.1k vs 19.0k (×.057). Mechanism: P(gold's window picked) =
.429; P(yes | window picked) = .294.

Verdicts (mechanical):
- **W1: PASS** — FLAT − WALK = +43.1 pts (flat does not lose).
- **W2: FAIL** — recall clause: WALK = 0.17 × FLAT (bar: ≥ 0.8).
  Token clause alone passed (×.058 ≤ ×.35); the AND fails.
- **W3: FAIL** — P(yes | window) .294 < 0.9 × FLAT recall = .466.
  The walk is diluted INSIDE the window as well as capped by its
  first step — not the P-series shape.

Post-hoc readings (not verdicts): the window selector is the
bottleneck — previews of one chunk's first 200 chars pick the right
window 43% of the time, and even with gold in scope the judge fires
at half its flat rate (window boundaries cut context; the menu's
0-option steals picks). The registered threshold sweep over the
persisted per-pair p_yes found an UNREGISTERED gem: early-stop over
the FLAT scan (judge chunks in doc order, stop when p_yes first
crosses t) reaches flat-parity recall (.517) at t=.5 judging a mean
4.7 chunks (≈ 1/10 of flat), and .50 at t=.6 with 3.2 chunks. This
is post-hoc — it was not registered as an arm and must be
re-registered before any use. Structure lost even where reads are
expensive; but an exit rule on the exhaustive scan buys most of
what the walk promised.

## J-series RESULTS (2026-10-14, scored — commit 19fe2a3, pinned corpus)

Run: pairscale.py --gen (464 unaudited train questions, seed base
20261013) then scored. Sanity: J1 all p3 .0690 = R run 2's c_p3
exactly. Pairs: J1 230 (58 questions), J8 1840 (464 questions).

| arm | direct | indirect | all (p1)  |
|-----|--------|----------|-----------|
| V0  | .207   | .138     | .172 (.069) |
| J1  | .069   | .069     | .069 (.000) |
| J8  | .069   | .069     | .069 (.000) |

Verdicts (mechanical):
- **JA: FAIL** — J8 − V0 = −10.3 pts (bar: ≥ +5 PASS, ≥ 0 PARTIAL).
- **JB: PASS (degenerate)** — J8 indirect .069 = J8 direct .069;
  the ratio holds at 1.0 because the head is uniformly at floor.
  Recorded as registered, read as no signal.

Post-hoc readings (not verdicts): 8× training data moved val BCE
.641 → .338 (the head fits the training distribution better) while
eval p3 stayed EXACTLY .069 and p1 exactly 0 — improved fit, zero
transfer. R2's FAIL was mechanism, not sample size: on [q; c; |q−c|;
q·c] features over this encoder, gold and same-doc distractors are
not separable at any tested scale. The cheap-head path on this
substrate is closed; the readout that works is the judge
itself.

## E-series (registered 2026-10-14, pre-data): a real RAG embedder

V-series closed every dial on the parasitic encoder (pooling,
prefixes, length — best +3.5 pts, no rescue). One escape hatch
remains, explicitly flagged in the V close-out: a DEDICATED
embedding model. Available substrate: Qwen3-Embedding-0.6B served
in a container (zean720, aksara-qwen3-embedding-1, gguf on GPU,
llama.cpp server, 1024-dim, L2-normalized, ctx 2048) — the model
family standard RAG stacks actually use.

Questions: the R-series frozen eval set (58), unchanged. Corpus:
pinned 5146e2d, unchanged. No judge calls, no training.

Arms (bi-embed + cosine; presence@1/@3 by recipe; ranking only):
- **V0** — the parasitic comparator, replicated from the slot-7
  cache (sanity: all-p3 within ±.03 of .172).
- **E0** — qwen3-embedding-0.6b, plain text both sides (query
  as-is, chunk as-is).
- **E1** — same model, query side wrapped in the model card's
  prescribed format: `Instruct: Given a question, retrieve
  passages that answer it\nQuery: {q}` (task string chosen a
  priori, generic); chunk side plain.

Predictions (mechanical):
- **EA (primary)**: best E-arm indirect p3 − V0 indirect p3
  ≥ +10 pts PASS (dedicated embedding rescues the comparator —
  R1's repair frame becomes re-askable); +3..+10 PARTIAL;
  < +3 FAIL (the task's geometry defeats small dedicated
  embedders too).
- **EB (instruct driver)**: E1 − E0 on indirect p3 ≥ +3 named
  "instruct helps"; ≤ −3 named "instruct hurts"; between →
  "neutral".
- **EC (chance separation)**: best E-arm all p3 ≥ 2 × random @3
  (= .256) PASS, else FAIL — the separation bar V0 never cleared
  robustly (.172).

Disclosures: substrate change for the retrieval arm only —
generation, judging, and every prior frozen artifact are
untouched (prior results remain model-bound to the 4B encoder;
E-series is a new column, not a re-scoring). Embeddings are
fetched from the REMOTE container over an SSH tunnel; a separate
cache file (embext_*.pkl) is never mixed with the slot-7 cache.
0.6B is the small sibling of the family — 4B/8B variants are
different arms, not tested here. The remote box runs shared
staging workloads; latency varies and is not a recorded metric.
Embedding content is the repo's own docs — no external data
leaves the lab beyond the tunnel to the operator's own host.

## E-series RESULTS (2026-10-14, scored — commits 5fa087b/registration, runner @ HEAD)

Run: extembed.py over an SSH tunnel to the container (fp fdf52bc3c7ec,
1024-dim, unit-norm). Sanity: V0 all p3 .1724 = V/R reference exactly.

Arms (presence@1/@3; judge column = R run 2 yes-set recall, a LOOSER
criterion than top-3 ranking — it counts over-collected yes-sets):

| arm                 | direct p3 | indirect p3 | all p3 (p1)   |
|---------------------|-----------|-------------|---------------|
| V0 (parasitic 4B)   | .207      | .138        | .172 (.069)   |
| E0 (qwen3-emb plain)| .724      | .552        | .638 (.448)   |
| E1 (instructed q)   | .724      | .517        | .621 (.414)   |
| J-llm flat judge    | .793      | .241        | .517 (—)      |

Per-doc p3 (E0): README .625, protocol .708, MS-137 .542, models 1.0 —
the 137-chunk doc where V0 scored .000 and the judge .625 is no longer
an inversion case.

Verdicts (mechanical):
- **EA: PASS** — best E-arm indirect p3 gap +41.4 pts (bar: ≥ +10).
  The dedicated embedder rescues the comparator outright; R1's repair
  frame is re-askable — and on the re-ask, the embedder beats the
  judge overall (.638 top-3 vs .517 yes-set recall) at ~zero
  marginal cost per query (cached corpus embeddings vs ~29k prompt
  tokens / ~19 s).
- **EB: instruct hurts** — E1 − E0 indirect p3 = −3.45 pts (direct
  unchanged). The model card's prescribed query-instruction format
  is a net negative on this corpus; registered a priori as a driver
  question, answered mechanically.
- **EC: PASS** — best E-arm all p3 .638 ≥ .256 (2 × random @3).
  Top-1 .448 ≈ 9× random @1 (.048).

Post-hoc readings (not verdicts): the judge keeps exactly one crown —
direct-question yes-set recall (.793 vs E0's .724 top-3). Everywhere
else the 0.6B embedder dominates, and the markedness story inverts:
the judge collapses on indirect questions (.241) while the embedder
holds .552 — the cheap readout is the ROBUST one. Every prior negative
about "the comparator" was a fact about the parasitic 4B encoder, not
about dense retrieval: V-series' close-out ("the encoder's geometry is
the ceiling") was correct, and the ceiling was the substrate, not the
method. Production implication for the sister repo (WireSense): dense
retrieval with this container is the obvious cheap-first stage, with
the judge reserved for high-precision direct confirmation — but any
such wiring is a new registered decision there, not a transfer made
here.
