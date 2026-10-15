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

## K-series (registered 2026-10-14, pre-data): decision layers on knowledge-graph QA

The one regime the lab never gave a fair shot: a structure too big
to read flat, with a GOOD cheap selector (E-series) instead of W's
text-preview garbage. Build a knowledge graph from the pinned
corpus, generate path-grounded questions, and ask whether a
decision layer traversing the graph beats retrieve-then-read at
matched token budget.

Substrate (built by registered procedure, committed before any
question is generated):
- **Graph**: for each of the 180 pinned chunks, the frozen model
  (slot 7, temp 0) extracts up to 6 triples {s, r, o} as JSON —
  entities as short noun phrases (≤ 4 words), relations as short
  verb phrases (≤ 3 words), explicit facts only. Deduped
  case-insensitively → results/kgraph.json (entities, triples,
  per-triple source chunk). Extraction noise is part of the
  substrate, disclosed, never hand-repaired.
- **Questions**: 60 eval, half 1-hop (a single triple), half 2-hop
  (a chain A–r1→B–r2→C present in the graph), each half split
  direct/indirect (R-series markedness recipes; temp .8, seeded).
  The generator sees the triples + their source snippets; gold =
  the terminal entity (canonical graph label). Eval is
  hand-audited BEFORE freeze (R-series policy; corrections
  recorded in an audit key). No train split — nothing trains.

Arms (all answer with an entity label; accuracy = normalized
match to gold; normalize = lowercase, collapse whitespace, strip
surrounding punctuation):
- **FLAT** — one call: the whole graph's triples + the question
  (anchor; expected to fit context on a lab-size graph).
- **READ** — embedder retrieval (E-series container via tunnel):
  rank entities by cosine to the query; expand 2 hops from the
  top-3 entries with per-hop breadth cap 12 (neighbors ranked by
  embedding); collect triples, re-rank by triple-text cosine,
  keep top 30; one LLM call over that subgraph.
- **WALK** — pure traversal: menu 1 picks among the top-5
  embedder-ranked entry entities; each later menu lists the
  current node's neighbors as `relation → object` (plus a stop
  option); the LLM picks; max 3 hops; the terminal node IS the
  answer. Entry asymmetry vs READ (top-5 menu vs top-3 expand)
  disclosed.
- **BEAM** — WALK keeping a frontier of 2: at each menu the top-2
  picks extend paths; path score = Σ (1st pick = 1.0, 2nd = 0.5);
  answer = best-scoring terminal (the lab's three-time replicated
  structural win, re-asked on graphs).

Metrics: accuracy by hops (1/2) and recipe (direct/indirect) and
all; prompt tokens per question per arm; hops taken; entry-hit
(gold path's start entity in the top-5 embedder entries).

Predictions (mechanical):
- **K1 (primary)**: WALK − READ on all-accuracy: ≥ 0 PASS (the
  decision layer beats retrieve-then-read at matched budget);
  within −2 pts PARTIAL; < −2 pts FAIL. A PASS/PARTIAL counts
  only if WALK tokens ≤ 1.1 × READ tokens (budget guard), else
  recorded FAIL-budget.
- **K2 (beam replication)**: BEAM − WALK ≥ +3 pts PASS, else
  FAIL; guard BEAM tokens ≤ 2.2 × WALK.
- **K3 (flat anchor)**: FLAT − best budgeted arm ≥ 0 PASS (flat
  stays supreme while the graph fits in context — the lab prior);
  a budgeted arm > FLAT + 2 pts FAIL (a first on this stack);
  else PARTIAL.

Disclosures: the graph is model-extracted from 180 chunks of the
operator's own docs — small (est. hundreds of triples), so FLAT
fitting in context is likely and K3's PASS would be the expected
null-anchor, not a finding about production-scale graphs; 2-hop
chains are sampled where the graph admits them (odd chains may
produce odd questions — the audit culls, pre-freeze); the embedder
is again the remote 0.6B container (E-series substrate) reached
over an SSH tunnel; WALK/BEAM menus are the same menu machinery as
R's J-menu (first-200-char previews not used — neighbor labels
only); temp-0 traversal is session-stable but not bit-deterministic
(W precedent).

Incident + Amendment (2026-10-14, before any K-series data): the
first scored attempt crashed on question 1 — FLAT's whole-graph
prompt measures ≈ 8.1k tokens of graph text alone, exceeding the
production server's per-slot context ceiling (65536 total across 8
slots → 8192 per slot); HTTP 400; the run wrote no output and no
metric was computed (only cache-only embeddings, kept). Amendment:
**FLAT = FLAT-2HALF** — the triple list is split deterministically
into two halves by index; each half is read flat with the same
question in one call; if the two answers conflict (both non-UNKNOWN
and different), one tie-break call picks between them; every read
counts toward FLAT's tokens. The arm still reads every triple with
no selection or ranking — the split is mechanical, disclosed here
before the rerun.

## K-series RESULTS (2026-10-14, scored — graph, frozen eval, and FLAT-2HALF amendment all pre-data)

Graph: 902 triples / 1352 entities extracted from the 180 pinned
chunks (frozen model, temp 0). Eval: 60 questions, 15 per
(hops × recipe) cell, hand-audited pre-freeze (43 generator keeps,
17 indirect rewrites against kept twin paths, 77 broken chains
dropped — the extraction noise the registration anticipated).
Entry-hit (gold path's start in the embedder's top-5 entities):
.533.

| arm        | all  | 1-hop | 2-hop | direct | indirect | tok/q |
|------------|------|-------|-------|--------|----------|-------|
| FLAT-2HALF | .317 | .267  | .367  | .433   | .200     | 10932 |
| READ       | .233 | .333  | .133  | .400   | .067     |   111 |
| WALK       | .150 | .200  | .100  | .267   | .033     |   412 |
| BEAM       | .283 | .367  | .200  | .467   | .100     |   705 |

Verdicts (mechanical):
- **K1: FAIL** — WALK − READ = −8.3 pts (bar: ≥ 0), and the budget
  guard also fails (WALK 412 tok vs READ 111, 3.7× the 1.1× cap):
  the greedy traversal decision layer loses to retrieve-then-read
  on both axes.
- **K2: PASS** — BEAM − WALK = +13.3 pts (bar: ≥ +3), tokens 705 ≤
  2.2 × 412 ✓. Beam-2's FOURTH consecutive replication (P depth
  extension, S routing, S replication, now graphs) and its largest
  margin yet.
- **K3: PASS** — FLAT − best budgeted = +3.3 pts (FLAT .317 vs
  BEAM .283). The flat anchor holds.

Post-hoc readings (not verdicts): the economics story this series
adds — BEAM keeps 89% of FLAT's accuracy at 6.5% of its tokens
(705 vs 10,932), the closest any budgeted arm has come to flat on
any substrate; on a graph too big to read (unlike this lab-size
one), beam-with-a-cheap-compass is the shape that survives. FLAT's
odd inversion (2-hop .367 > 1-hop .267) contradicts every other
arm and is likely the half-split + tie-break interacting with
1-hop golds that include degenerate entity labels (the audit kept
some terse ones deliberately). READ's indirect collapse (.067)
echoes R: graph labels do not paraphrase — indirect needs the
original chunk text, which extraction destroys. Absolute
accuracies are low across all arms (.15–.32): the extracted graph
is lossy substrate, and entry-hit .533 says short entity labels
embed far worse than full chunks did (.638 top-3 in E-series).
The lab's through-line holds: greedy structure loses (K1), the
one structural pattern that pays is keeping candidates alive (K2),
and flat still wins when it fits (K3).

## K-series Amendment 2 (2026-10-14, pre-data): LOOKAHEAD arms

K1's diagnosis: greedy walkers are myopic — they pick one edge and
discover too late where it led. The operator proposes the fix:
choose a whole ROUTE at a time. Registered here before any data:

- **LOOK-1** (the proposal, as stated): entry = same top-5 entity
  menu as WALK/BEAM (held constant across traversal arms). From
  the current anchor, enumerate simple paths (no repeated
  entities) up to 3 hops; the LLM sees the numbered path list and
  picks ONE (or 0 = stop → answer is the current anchor). The
  chosen path's far endpoint becomes the new anchor; repeat for
  at most 2 segments (the walker cannot see the destination, so
  "until reached" is operationalized as a stop option + segment
  cap). Answer = the final chosen path's terminal node — the same
  endpoint scoring as WALK/BEAM for comparability.
- **LOOK-2**: identical, but the entry menu keeps TWO starting
  entities (weights 1.0/0.5, K2 precedent) and each segment keeps
  a frontier of 2 best-scoring routes; answer = the best-weight
  route's terminal.
- Baselines FLAT/READ/WALK/BEAM are NOT rerun — their per-question
  records are committed (kgraphqa.json, same frozen eval, same
  graph); the comparison reads them. New results go to a NEW file
  (klookahead.json); nothing is overwritten.

Anti-explosion mechanism (registered — "all paths" is exponential,
hubs in this graph touch dozens of triples): at each hop extension
keep only the top-12 incident triples by embedder cosine of the
TRIPLE TEXT to the question (all triple embeddings are already
cached from K; ranking adds zero new embeds); intermediate pools
capped at 300 (2-hop) / 800 (3-hop); the LLM menu shows the top
60 paths ranked by mean triple cosine. Entry procedure, stop
option, weights, and endpoint scoring are identical to the K
traversal arms.

Predictions (mechanical):
- **KL1 (primary)**: LOOK-2 all-accuracy − BEAM all-accuracy
  ≥ +3 pts AND LOOK-2 tokens ≤ 2 × BEAM tokens → PASS (lookahead
  adds to the law); ≥ +3 with budget blown → PASS-budget; 0..+3 →
  PARTIAL; < 0 → FAIL (route-level choice adds nothing over edge
  hedging).
- **KL2**: LOOK-1 − WALK ≥ 0 → PASS (lookahead rescues greedy at
  the route level); else FAIL.
- **KL3 (descriptive, no verdict)**: rate of questions where gold
  sits ON the chosen route but is not its terminal (mid-path
  answers the endpoint scoring misses).

Disclosures: the 60-path menu cap and 2-segment cap are the
registered budget; enumeration ranking is embedder-based (the
graph-label embedding weakness measured in K applies); entry
top-5 hit rate is .533 (K-measured ceiling on any traversal);
baselines are reused cross-run, not re-measured (they are
deterministic-temp records of the same frozen questions and
graph; W-series precedent is a same-session rerun for the judge —
not needed here, no new judging of baselines).

## K-series Amendment 2 RESULTS (2026-10-14, scored — klookahead.json, baselines read from committed kgraphqa.json)

| arm    | all  | 1-hop | 2-hop | direct | indirect | tok/q |
|--------|------|-------|-------|--------|----------|-------|
| LOOK-1 | .100 | .133  | .067  | .133   | .067     |   406 |
| LOOK-2 | .033 | .033  | .033  | .033   | .033     |   825 |
| (WALK) | .150 | .200  | .100  | .267   | .033     |   412 |
| (BEAM) | .283 | .367  | .200  | .467   | .100     |   705 |

Verdicts (mechanical):
- **KL1: FAIL** — LOOK-2 − BEAM = −25.0 pts (bar: ≥ +3). Route-level
  choice does not add to the hedge law; at a frontier of 2 it
  collapses below everything (2/60).
- **KL2: FAIL** — LOOK-1 − WALK = −5.0 pts (bar: ≥ 0). Lookahead
  does not rescue greedy walking — it deepens it (the walker now
  commits to THREE edges at once instead of one).
- **KL3 (descriptive)**: gold sits ON the chosen route 31.7% of
  the time, and in essentially all of those cases the terminal is
  wrong (on-route-but-terminal-wrong = 30.0%). The walker reaches
  the answer's neighborhood and then walks PAST it: a 2-hop gold
  forces a 3-hop route whose endpoint is one hop beyond the gold.

Post-hoc readings (not verdicts): the design's fate was sealed by
two compounding mechanisms. (1) Committing at the ROUTE level is
still committing — Amendment 2's registered alternative (frontier
of 2) inherited the law's benefit in no measure, because route
commitment multiplies the overshoot instead of hedging it: more
route = more terminal-past-gold. (2) The scoring contract (terminal
entity, chosen for comparability with WALK/BEAM) punishes exactly
the success mode lookahead produces — passing THROUGH the answer.
A mid-route scoring arm (any node on the traversed route counts)
is the obvious re-registration; it would need to apply to WALK and
BEAM as well for fairness, and until then no claim transfers about
lookahead being worse than beam "at answering" — only at this
endpoint contract, where it is much worse. The operator's design
instinct was half right: the diagnosis (myopia) was the correct
failure mode; the fix that survives measurement is not longer
committed routes but wider hedges.

## X/Y/K3/Z series (registered 2026-10-14, pre-data): the follow-up batch

Four registrations, committed before any of their data exists.

**Correction (disclosed here first):** the W-series RESULTS prose
described its post-hoc sweep as "judge chunks in doc order, stop
when p_yes first crosses t". The committed walkret.py sweep
actually computed an ORDER-FREE quantity — P(p_yes(gold) ≥ t) plus
the mean yes-set size at t. The "1/10 of the cost" reading
therefore assumed a scan that was never simulated. X-series runs
the real sequential procedure.

### X-series: early-stop judging on embedder order

Questions: the frozen 58 (R/E/W eval set). Corpus: pinned chunks.
Judge: CR.judge (same prompt, temp 0, slot 7). Chunk order for the
embedder arms: E0 cosine (committed cache; zero new embeds).

Arms (per question):
- **X-ES** — scan chunks in EMBEDDER rank order; judge each; stop
  at the first chunk with p_yes ≥ τ (τ = .5, chosen a priori from
  the W sweep, disclosed); scanned set = judged prefix incl. stop
  chunk.
- **X-DS** — identical with DOC order (the naive baseline).
- Baselines (committed, not rerun): flat yes-set recall .5172 and
  flat tokens 29,236/q (walkret.json; its r2 agreement was 1.0).

Metrics: **stop-recall** (gold IS the stop chunk — the deployment
answer set), **prefix-recall** (gold anywhere in the scanned
prefix), chunks judged, prompt tokens. Sanity: per-judged-chunk
p_yes agreement with the committed flat p_yes (flag < .90;
temp-0 cross-session precedent is 1.0).

Predictions (mechanical):
- **XA**: X-ES prefix-recall ≥ 0.95 × .5172 AND mean tokens ≤
  0.25 × 29,236 → PASS; else FAIL.
- **XB**: X-ES stop-recall ≥ X-DS stop-recall → PASS (embedder
  order beats doc order where it matters); else FAIL.
- **XC (descriptive)**: mean chunks judged (both arms), gold's
  mean embedder rank, prefix-vs-stop gap.

### Y-series: query rewriting for indirect questions

The indirect collapse is the lab's most repeated gap (judge .241,
graph-READ .067, embedder .552 vs direct .724). Arm **Y-RW**: one
LLM call (temp 0, seeded, slot 7) rewrites each frozen question
into documentation vocabulary; the E0 embedder ranks chunks
against the REWRITTEN query. Rewriter sees ONLY the question —
no corpus content (no leakage; prompt registered in the runner).
Baseline: committed extembed.json E0 per-question records
(index-aligned, same frozen order).

Predictions (mechanical):
- **YA**: Y-RW indirect p3 − E0 indirect p3 (.552) ≥ +5 pts PASS;
  0..+5 PARTIAL; < 0 FAIL (rewriting hurts).
- **YB (direct guard)**: Y-RW direct p3 − E0 direct p3 (.724)
  ≥ −3 pts PASS, else FAIL.
- Descriptive: fraction of rewrites identical to the original.

### K-series Amendment 3: mid-route scoring

KL3 showed traversal routes reach the gold's neighborhood and the
endpoint contract punishes it. Re-run ALL FOUR traversal arms
(WALK, BEAM, LOOK-1, LOOK-2) on the frozen 60 with persisted
routes; score **mid-route**: correct iff gold ∈ route nodes (any
position). FLAT/READ have no routes — their committed endpoint
numbers stand as the comparator. Sanity: fresh endpoint scoring
vs committed endpoint per arm (flag |Δ| > .03, session-stability
note, not verdict-gating).

Prediction (mechanical):
- **KM1**: BEAM-mid(all) − READ-endpoint(all) (.2333 committed)
  ≥ 0 → PASS (under the fair contract, hedged traversal matches
  retrieve-then-read); else FAIL.
- **KM2 (descriptive)**: per-arm mid-vs-endpoint deltas; the
  LOOK overshoot share under mid scoring.

### Z-series: temperature calibration of judge confidence

Data disclosure: the underlying per-pair scores ARE committed
(walkret.json flat p_yes for all 58 × their chunks). The NEW data
is the calibration fit and its held-out evaluation, which does
not exist yet. Split: seeded 29 fit / 29 holdout questions.
Fit: one temperature T on the logit (p' = σ(logit(p)/T))
minimizing NLL on fit-slice pairs (label = chunk is gold).
Evaluate on holdout: ECE (10 bins) raw vs calibrated.

Predictions (mechanical):
- **ZA**: holdout ECE_calibrated < holdout ECE_raw − 0.01 →
  PASS; |Δ| ≤ 0.01 PARTIAL; else FAIL (calibration is not the
  missing piece).
- **ZB (stability)**: T ∈ [0.5, 2.0] and the ECE improvement
  direction agrees between fit and holdout slices → PASS, else
  FAIL.
- Descriptive: the τ that achieves ≥ 90% of flat recall on fit
  vs holdout (does the stop-bar transfer).

## X/Y/K3/Z RESULTS (2026-10-14, scored — one batch, all registered pre-data)

### X-series RESULTS (xstop.json)

Sanity: per-judged-chunk p_yes agreement with committed flat
scores = 1.000 (temp-0 judge reproduced every decision). Gold's
mean embedder rank: 6.5.

| arm | prefix | stop | chunks judged | tok/q |
|-----|--------|------|---------------|-------|
| X-ES (embedder order) | **.690** | .397 | 11.5 | 4,615 (×.158) |
| X-DS (doc order)      | .621 | .310 | 32.0 | 12,805 (×.438) |
| (flat committed)      | .517* | .517* | 180 | 29,236 |

*flat's .517 is yes-set recall (a different answer-set contract —
see below).

Verdicts: **XA PASS** (prefix .690 ≥ .491 bar; tokens ×.158 ≤ ×.25)
— the sequential scan BEATS the exhaustive yes-set baseline at 16%
of the cost. **XB PASS** (stop-recall .397 vs .310, +8.6 pts):
embedder order beats doc order where deployment lives.

Post-hoc (not verdicts): prefix (.690) > flat yes-set (.517)
because the contracts differ — the scan counts gold anywhere it
reads, flat only when the judge says yes. The deployment-honest
number is stop-recall .397: one answer, 4.6k tokens, vs flat's
3.4-average-size yes-set at 29.2k. The embedder order is doing
real work: gold sits at mean rank 6.5, so the scan reaches the
answer's neighborhood in the first dozen reads.

### Y-series RESULTS (qrewrite.json)

Rewriting happened (only 3.4% of rewrites identical to the
original) and it HURT: Y-RW indirect p3 .483 vs E0 .552 (−6.9),
direct .690 vs .724 (−3.45). Verdicts: **YA FAIL, YB FAIL** (the
direct guard also tripped). The 0.6B embedder prefers the user's
raw question; rewriting is the second measured anti-lesson about
"helping" the query (after E-series' instruct prefixes).

### K-series Amendment 3 RESULTS (kmid.json)

Endpoint drift of the fresh run vs committed records: walk 0.000,
beam 0.000 (session-perfect).

| arm (mid-route scoring) | mid  | end (fresh) | end (committed) | tok/q |
|--------------------------|------|-------------|-----------------|-------|
| LOOK-2                   | **.333** | .033 | —     |   825 |
| BEAM                     | .300 | .283        | .283            |   705 |
| LOOK-1                   | .283 | .100        | —               |   406 |
| WALK                     | .250 | .150        | .150            |   412 |
| (READ, endpoint)         | —    | —           | .233            |   111 |

Verdicts: **KM1 PASS** — BEAM-mid .300 ≥ READ-endpoint .233
(+6.7 pts): under the fair contract, hedged traversal beats
retrieve-then-read on this graph. **KM2 (descriptive)**: the
mid-vs-endpoint delta is the LOOKAHEAD diagnosis quantified —
LOOK-2 gains +30 pts when routes get credit (.033 → .333): its
routes pass through the gold a third of the time and its endpoint
contract threw that away. Under mid scoring the operator's
route-lookahead design is the best graph number in the lab
(.333), ahead of beam (.300) — at 825 tok/q with two frontiers.

Caveat (disclosed): mid-route scoring credits a walk that merely
brushes the gold node en route to somewhere else; with 2-hop gold
paths and 3-hop routes, partial credit is real information but not
answer-graded. The registered claim stands as KM1; LOOK-2's .333
is the headline descriptive.

### Z-series RESULTS (calib.json)

T = 0.66 (the judge is overconfident; sharpening was needed, not
softening — p' = σ(logit p / 0.66)). ECE: fit .092 → .065,
holdout **.105 → .081**. Verdicts: **ZA PASS** (−.024, bar −.01),
**ZB PASS** (T in range, direction consistent). Descriptive: the
τ that achieves ≥ 90% of flat recall is .05 on BOTH slices — the
stop-bar transfers perfectly, so X-series' τ = .5 was a
conservative choice, not a tuned one.

### Batch verdict (the through-line, updated)

The production shape this lab now endorses, all measured: **rank
with the dedicated embedder (E), scan-and-stop in that order with
the judge as verifier (X), calibrate the stop bar on held-out
data (Z), and skip query rewriting (Y).** On graphs: hedge, never
commit (K + Amendment 2), score what the walker actually visits
(Amd 3), and expect the flat anchor to win whenever the graph
fits (K3).

## A-series registration: end-to-end assembly (2026-10-14, registered pre-data)

Every component of the endorsed recipe has passed separately —
embedder ranking (E), embedder-ordered scan-and-stop (X), calibrated
bar (Z) — but the assembled machine has never been scored as one
piece. This closes the loop: one frozen question set, three arms,
answers generated and graded end-to-end.

**Procedure** (per frozen eval question, n=58, r_questions.json;
corpus pinned 5146e2d; frozen model slot 7, temp 0; ext-embedder
cache fp fdf52bc3c7ec, verified complete pre-run — no tunnel):

1. Rank the document's chunks by ext-embedder cosine to the question.
2. Scan in that order, judging each chunk (CR.judge instrument).
   Stop bar: first calibrated p ≥ .5, with p' = σ(logit p / 0.66)
   (T committed in calib.json). **Disclosed**: at τ = .5 the
   calibrated bar equals the raw bar (σ fixed point) — calibration
   is carried as recorded procedure (Z), not re-tested here; a lower
   calibrated operating point is future work.
3. **Fallback**: if no chunk crosses the bar, answer from the
   embedder top-1 chunk (the librarian's pick) rather than returning
   nothing. This is the deployment answer to X's stop-vs-prefix gap.
4. Generate: system "Answer using only the excerpt. One sentence,
   no preamble."; user "Question: {q}

Excerpt:
{chunk}";
   temp 0, max_tokens 128, seed 20261014 + 17·i (i = question index).
5. Grade: system "You are a strict grader. Answer with exactly one
   word: yes or no."; user "Reference passage:
{gold chunk}

   Question: {q}

Proposed answer: {ans}

Does the proposed
   answer correctly answer the question, as supported by the
   reference passage?"; first-token yes/no mass, correct ≡ p ≥ .5.
   **Disclosed**: the grader is the same frozen model that generates —
   judge-graded answers are the lab's instrument style throughout,
   not human verification.

**Arms** (all in-run, same 58 questions):
- TOP1 — pure RAG: answer from the embedder top-1 chunk. No
  decision layer.
- FLATPOL — the X-DS policy at answer level: scan in document order,
  stop at first raw p ≥ .5, fallback chunk 0, answer from the
  stopped chunk.
- ASM — the machine: steps 1–5 (embedder order + fallback).

**Predictions (registered bars)**:
- **A1 (headline)**: ASM answer accuracy ≥ TOP1 + .03 → PASS;
  |Δ| ≤ .03 → PARTIAL; else FAIL. The machine must beat pure
  vector RAG, not tie it.
- **A2 (ordering earns its keep at answer level)**: ASM ≥ FLATPOL →
  PASS; within −.02 → PARTIAL; else FAIL.
- **A3 (stop-chunk quality)**: ASM stopped-chunk-hit (gold equals
  the answered chunk, fallback hits counting) ≥ committed X-DS
  stop-recall .3103 → PASS; else FAIL.
- **A4 (budget)**: ASM total tokens/question (scan + generate +
  grade) ≤ .35 × committed FLAT_TOK 29,236 = 10,233 → PASS-budget;
  else FAIL-budget.
- Sanity (disclosed field, not a verdict): per-judged-chunk p_yes
  agreement with committed walkret.json flat scores ≥ .90 (X's
  instrument). Smoke runs marked and never scored.

Descriptive (registered, no bar): per-recipe answer accuracy per
arm, fallback trigger rate, judge/generator agreement (does the
grader say yes exactly when the scanner stopped yes).

## A-series RESULTS (assembly.json, 2026-10-14 — scored)

Sanity: per-judged-chunk p_yes agreement with committed flat
scores = 1.000 (temp-0 judge reproduced every committed decision,
third consecutive session).

| arm | answer acc | direct | indirect | answered-chunk hit | tok/q |
|-----|-----------|--------|----------|--------------------|-------|
| ASM (the machine)   | **.431** | .655 | .207 | **.466** | 5,428 |
| FLATPOL (doc order) | .379 | .621 | .138 | — | 13,640 |
| TOP1 (pure RAG)     | .345 | .517 | .172 | — | 798 |

Verdicts: **A1 PASS** (+.086 over pure vector RAG, bar +.03) —
the assembled machine beats its own strongest single component at
answer level. **A2 PASS** (+.052): embedder ordering earns its keep
under an identical policy — the ordering is the active ingredient,
not the scan-and-stop shape. **A3 PASS** (+.155): answered-chunk
hit .466 vs the committed X-DS anchor .310 — and +6.9 pts over
X-ES's own .397 stop-recall, the fallback converting part of X's
stop-vs-prefix gap into actually-answered chunks. **A4
PASS-budget**: 5,428 tok/q = 18.6% of the exhaustive scan's 29,236
(half of FLATPOL's 13,640). With this, every registered prediction
of the batch passed: the recipe is a scored machine, not a parts
list.

Descriptives (registered): the fallback fired on 12/58 questions
(20.7%); its answers grade lower (mean grade_p .348) than
scan-confirmed answers (.472) — grader confidence tracks the
scanner's decision, as it should. Answer synthesis still loses
~3.4 pts with the right chunk in hand (.466 hit vs .431
graded-correct). And the indirect cell remains the frontier: .207,
with nearly all of the machine's edge over TOP1 coming from the
direct cell (+13.8 vs +3.5).

## B-series (registered 2026-10-15, pre-data): the fair fight — machine vs production-style top-k RAG

A-series closed with a declared loophole: its pure-RAG arm (TOP1,
.345) is the strictest variant — one chunk, no k. Production RAG
stacks feed top-k (3–5) excerpts in a single context. The committed
E-series numbers say this fight is live and unwon: E0 top-3 context
presence is .638 — ABOVE the machine's answered-chunk hit .466 — so
any answer-synthesis loss under ~.19 on the extra context hits and
the plain baseline wins on paper. The A-series endorsement ("the
machine beats its own components") has never been scored against
the real competitor. This series scores that fight.

Questions: frozen 58, unchanged. Corpus: pinned 5146e2d,
unchanged. Ranking: E0 (qwen3-embedding-0.6B, plain query, fp
fdf52bc3c7ec, cache complete — no tunnel; identical ranking code
path to A-series TOP1). Generator and grader: frozen slot-7 model,
prompts/seeds identical to A-series (disclosed there: grader =
generator; temp 0; answer seed 20261014+17·i per question index).

Arms (per frozen eval question):
- **TOPK-k, k ∈ {2, 3, 5, 8}** (k_eff = min(k, n_chunks)) —
  context = top-k chunks in E0 order, concatenated in ONE prompt:
  `Question: {q}\n\nExcerpt 1:\n{c1}\n\nExcerpt 2:\n{c2} ...`
  system = A-series GEN_SYS verbatim; generate (temp 0, max 128,
  A-series seed); grade vs the gold chunk with the A-series grader
  verbatim. No fallback, no verification — production RAG answers
  from whatever the retriever returns.
- **ASM — not re-scored**: per-question records load from
  committed assembly.json (same seeds, same prompts → exact
  pairing). Its committed budget anchor for this series:
  5,427.7 tok/q.
- **TOP1 re-run** — the A-series TOP1 arm re-executed in this
  session as the determinism sanity (below).

Cost accounting: answer prompt + grade prompt tokens (the
A-series counter). Embedding marginal cost ≈ 0 for every arm
(cache). Expected TOPK tokens: single-excerpt answer+grade ≈ 800
⇒ k=8 ≈ 3–4k — ALL swept k expected under the machine's budget,
so B2 below is expected to demand the machine beat even top-8.

**Bars (mechanical)**:
- **B1 (production-default RAG)**: ASM answer accuracy ≥ TOPK-3 →
  PASS; −.02 ≤ Δ < 0 → PARTIAL; else FAIL. The machine must beat
  RAG as actually deployed, not RAG handicapped to k=1.
- **B2 (budget dominance)**: let S = {swept k : mean tok(TOPK-k) ≤
  committed ASM 5,427.7}. If S ≠ ∅: Δ = ASM − max(TOPK-k accuracy
  over k ∈ S) — the machine must beat the BEST affordable top-k.
  PASS Δ ≥ 0; PARTIAL ≥ −.02; else FAIL. (S is selected by a rule
  fixed here, evaluated on measured tokens.)
- Sanity (disclosed field, not a verdict): TOP1 re-run per-question
  answer-string AND picked-chunk agreement vs committed
  assembly.json ≥ .95 (temp-0 determinism, fourth consecutive
  session). Smoke runs marked and never scored.

**Registered bets (direction, pre-data)**:
- **PB1: B1 expected FAIL** — TOPK-3 > ASM, predicted Δ +.03..+.15
  (context ceiling .638 vs .466; k=1 synthesis loss only −3.4 pts;
  dilution would have to exceed −.19 on the extra hits to save the
  machine).
- **PB2: B2 expected FAIL** — the best affordable k (likely 5)
  beats the machine; accuracy non-monotone in k (rises to k=3,
  sags by k=8 as dilution grows — Helping-Hurts, law 06, at answer
  level).
- **PB3: sanity PASS** (determinism).
- Descriptive bets (no bar): the indirect cell flips to RAG (E0
  indirect p3 .552 vs machine indirect answer .207); tokens grow
  ≈ linearly in k; the machine's surviving claim, if any, shifts
  from raw accuracy to verified provenance — it answers only from
  a chunk the judge confirmed above bar, vs RAG's unconditional
  answer.

## B-series RESULTS (topk.json, 2026-10-15 — scored)

Sanity: the TOP1 re-run reproduced committed assembly.json on all
58 questions — answer strings AND picked chunks, agreement 1.000,
headline drift 0.000 (fourth consecutive session of temp-0
determinism).

| arm | ctx hit | answer acc | synthesis (acc given hit) | tok/q |
|-----|---------|-----------|---------------------------|-------|
| TOP1 (k=1, committed)       | .448 | .345 | .77  | 798 |
| TOPK-2                      | .552 | **.397** | .625 | 1,146 |
| TOPK-3 (production default) | .638 | .345 | .514 | 1,467 |
| TOPK-5                      | .759 | .328 | .432 | 2,089 |
| TOPK-8                      | .810 | .345 | .425 | 3,045 |
| ASM (the machine, committed)| .466 (answered) | **.431** | **.852** | 5,428 |

Verdicts: **B1 PASS** (+.086 over production-default top-3 RAG —
which scored exactly TOP1's .345: +19 pts of context hit bought
zero end-to-end accuracy, and the A1/B1 margins coincide to four
decimals for that reason). **B2 PASS** (+.034): as predicted, ALL
swept k came in under the machine's budget, and the machine beat
the best of them (top-2, .397) — paired record 8 wins / 6 losses /
44 ties.

The registered bets FAILED in direction, and the failure is the
finding. PB1/PB2 predicted top-k RAG wins (Δ +.03..+.15); the
machine passed both bars instead. The mechanism bet was right —
dilution — but its magnitude was underestimated: synthesis given
gold-in-context falls monotonically .770 → .625 → .514 → .432 →
.425 across k = 1→8 (mean grade_p falls in lockstep .425 → .392 →
.361 → .384), almost exactly cancelling the context-hit gain
.448 → .810. Answer accuracy is a flatline (.33–.40) across the
entire sweep: every excerpt the retriever adds is worth as much
dilution as recall. Law 06 (helping-the-input-hurts), now measured
at answer level with a full dose–response curve.

Where the machine's edge lives: it synthesizes from ONE
judge-verified chunk (.852 of answered hits) vs .43–.63 for any
multi-excerpt dump. In the direct cell the gap is stark: at k=8
gold is in the context for 96.6% of direct questions and RAG still
answers only .552 — the machine's .655 comes from reading one
confirmed page, not eight candidates. The indirect cell flips to
RAG, as betted descriptively (.241 at k=2 vs .207) — the frontier
is unchanged.

Honest cost note (disclosed, not a bar): the machine did NOT win
on price. It spent 5,428 tok/q; top-2 RAG spent 1,146 and kept 92%
of the machine's accuracy. The registered claim was accuracy
dominance over everything at-or-under its budget — that holds. The
per-token efficiency crown stays with plain top-k RAG; the
machine's crown is the most accuracy this lab knows how to buy
without the 29,236-token exhaustive read.

Tally: 18 of 34 headline predictions passed.

## C-series (registered 2026-10-15, pre-data): the pool-confined machine — RAG retrieves, the machine reads

B-series resolved the division of labor: the top-8 pool holds the
gold page .810 of the time (vs the machine's .466 answered-chunk
hit) but dilution kills pile synthesis, while the machine's
verified-single-page reader synthesizes .852 — at 5,427.7 tok/q.
The combination question: can the top-8 pool host the machine's
scan-and-stop, buying the pool's recall position at a fraction of
the full-document scan cost?

**Disclosed post-hoc simulation (committed records only, no new
model calls — this is the pre-registered quantitative prediction).**
Method: committed walkret.json raw p_yes per chunk + committed
topk.json top-8 picks (the frozen B artifact); calibrate at T=0.66
(committed calib.json); walk the pool in embedder order, stop at
the first cal p ≥ .5, else answer from rank-1. Results (chunk
level, n=58): pool-8 stop-hit .4655 — IDENTICAL to the committed
machine's .4655 (42/46 live stops were already inside the pool;
the 4 deep stops yielded 1 hit, recovered by the rank-1 fallback
in simulation) — fallback firing 16/58, mean 3.6 judged pages
(~2.2k tok/q projected). Pool-4 ties (.4655, 2.1 judged). Two
rejected variants, simulated: max-p fallback (answer from the
pool's highest below-bar score) FAILS (.3793 — below the bar the
judge's scores do not rank truth; the confidently-wrong page beats
the under-scored right page, so the fallback stays rank-1: rank is
the only signal that survives below the bar); SET-8 (judge all 8,
answer from best yes) ties at higher cost — dominated.

Questions: frozen 58. Corpus: pinned 5146e2d. Judge/generator/
grader: frozen slot-7 model, prompts and seeds identical to
A/B-series (grader = generator, disclosed there; answer seed
20261014+17·i). Pool: loaded from committed topk.json topk8.pick
per question (no re-embedding).

Arms (per frozen eval question):
- **ASM-8 (primary)** — scan the committed top-8 pool in embedder
  order with the live judge; stop at first cal p ≥ .5; answer from
  that ONE page; fallback rank-1 if nothing crosses. Identical
  GEN_SYS/grade instruments to A/B-series.
- **ASM-4 (secondary, descriptive)** — same, confined to the first
  4 pool pages; judgments shared with ASM-8 (prefix), answered
  separately when the decision differs. No bar; pool-size
  dose–response at answer level.

**Bars (mechanical)**:
- **C1 (non-inferiority)**: ASM-8 answer accuracy ≥ committed ASM
  .4310 − .02 → PASS; ≥ −.04 → PARTIAL; else FAIL.
- **C2 (cost)**: ASM-8 total tokens/question ≤ .5 × committed
  5,427.7 = 2,713.8 → PASS-budget; else FAIL-budget.
- **C3 (still beats RAG)**: ASM-8 ≥ committed TOPK-2 .3966 → PASS;
  ≥ −.02 → PARTIAL; else FAIL.
- Sanity (disclosed field, not a verdict): per-judged-chunk
  live-vs-committed stop-decision agreement (the A-series
  instrument, committed at 1.000) ≥ .95; live answered-chunk
  agreement with the disclosed simulation reported alongside.
  Smoke runs marked and never scored.

**Registered bets (pre-data, from the disclosed simulation)**:
- **PC1**: live ASM-8 answered-chunk hit within ±1 question of the
  simulated 27/58 (.4655).
- **PC2**: fallback count within ±2 of the simulated 16/58.
- **PC3**: mean judged pages within [.2, .5] of the simulated 3.6.
- **PC4 (the risky one)**: answer-level synthesis holds up on the
  newly answered chunks — ASM-8 ans_acc ≥ .38, C1 PASS.
- Descriptive bets (no bar): fallback answers again grade lower
  than stopped answers (A-series pattern); ASM-4 within a question
  of ASM-8 at the chunk level; the capture gap (pool has gold
  .810, machine lands .466) persists — the verifier, not the
  retriever, remains the binding constraint.

## C-series RESULTS (poolstop.json, 2026-10-15 — scored)

Sanity: per-judged-chunk stop-decision agreement with committed
walkret.json p_yes = 1.000 (fifth consecutive session of temp-0
determinism), and the live run reproduced the disclosed simulation
EXACTLY — answered-chunk agreement 1.000 on both arms. Every
registered bet landed: stop-hit 27/58 = .4655 (simulated .4655),
fallback 16/58 (simulated 16), mean judged 3.672 (simulated 3.6),
answer level .4138 ≥ .38 with C1 PASS.

| arm | answered hit | answer acc | direct | indirect | judged/q | tok/q |
|-----|--------------|-----------|--------|----------|----------|-------|
| ASM (committed, full doc) | .466 | **.431** | .655 | .207 | 11.5 | 5,428 |
| **ASM-8 (pool-confined)** | .466 | .414 | .621 | .207 | 3.67 | **2,206** |
| ASM-4 (pool-confined)     | .466 | .379 | .552 | .207 | 2.41 | 1,732 |
| TOPK-2 (best RAG, committed) | — | .397 | .552 | .241 | 0 | 1,146 |

Verdicts: **C1 PASS** (−.0172, inside the ±.02 non-inferiority
band), **C2 PASS-budget** (2,206.2 tok/q = 40.6% of the machine's
cost, vs the 2,713.8 ceiling), **C3 PASS** (+.0172 over the best
RAG). The pool-confined machine is scored: same answered-chunk
hit (.466), same indirect cell (.207), one direct answer fewer,
at 2.5× less money — and still ahead of every RAG arm.

The one lost question, dissected (it is the entire −.0172): the
question's gold page sits at embedder rank ~13, OUTSIDE the top-8
pool, and the judge crosses the bar only there — the committed
machine found it after 14 judgments; the confined machine
exhausted the pool (max cal p .249) and fell back to rank-1,
answering wrong. The chunk-level ledger balances (−1 deep hit, +1
extra rank-1 fallback hit — both arms land 27/58), but fallback
answers are the weak link (ans_ok .125 vs .524 for stopped
answers; grade_p .312 vs .486 — the A-series pattern, louder), so
the trade shows at answer level. Paired against the committed
machine, this is the ONLY question either way.

ASM-4 ties ASM-8 at the chunk level (.466, as simulated) but pays
at answer level: five extra fallbacks cost two answers
(.379 vs .414) — the dose–response lives in fallback quality, not
in hits. The capture gap is untouched, as registered: the pool
holds gold .810 of the time and the verifier still lands .466 —
the judge, not the retriever, is now the binding constraint.

Deployment shape after three series: rank with the dedicated
embedder, scan the top-8 with the calibrated judge, stop at the
first confirmed page, answer from that page alone, fall back to
rank-1 — .414 at 2,206 tok/q (7.5% of the exhaustive read), vs
.345–.397 for any form of pile RAG.

Tally: 21 of 37 headline predictions passed.

## S-amendment (registered 2026-10-15, pre-computation): paired significance for the answer-level season

Every headline gap so far is a point estimate on n=58. Before any
further interpretation — and before any new scoring — the season's
six paired gates get mechanical uncertainty. Method (fixed here,
computed by stats.py from committed records only, no model calls):
exact McNemar (two-sided binomial on the discordant pairs) plus a
paired bootstrap (10,000 resamples, seed 0, percentile 95% CI) on
the accuracy difference.

Comparisons (paired per question; ASM = committed assembly.json):
- **S-A1**: ASM vs TOP1 (A1's +.086)
- **S-A2**: ASM vs FLATPOL (A2's +.052)
- **S-B1**: ASM vs TOPK-3
- **S-B2**: ASM vs TOPK-2
- **S-C1**: ASM-8 vs ASM (the −.0172 non-inferiority; registered
  question: is the data consistent with a loss beyond −.02?)
- **S-C3**: ASM-8 vs TOPK-2

Interpretation (registered, mechanical): **SOLID** = McNemar
p < .05 AND the bootstrap CI excludes 0 in the registered
direction; **SUGGESTIVE** = p < .10; **NOT ESTABLISHED**
otherwise. For S-C1 specifically: **SOLID-NI** = the CI's lower
bound ≥ −.02 (even the pessimistic end stays inside the
registered band). This amendment ANNOTATES, never re-adjudicates:
no prior verdict changes either way.

## S-amendment RESULTS (stats_amendment.json, 2026-10-15 — computed)

No model calls; committed per-question records only.

| comparison | diff | discordant a/b | McNemar p | 95% CI | read |
|---|---|---|---|---|---|
| S-A1 ASM vs TOP1      | +.086 | 6/1 | .125 | [−.086, +.259] | NOT ESTABLISHED |
| S-A2 ASM vs FLATPOL   | +.052 | 7/4 | .549 | [−.121, +.224] | NOT ESTABLISHED |
| S-B1 ASM vs TOPK-3    | +.086 | 8/3 | .227 | [−.086, +.259] | NOT ESTABLISHED |
| S-B2 ASM vs TOPK-2    | +.034 | 8/6 | .791 | [−.155, +.207] | NOT ESTABLISHED |
| S-C1 ASM-8 vs ASM     | −.017 | 0/1 | 1.0  | [−.190, +.155] | NOT ESTABLISHED (NI not established) |
| S-C3 ASM-8 vs TOPK-2  | +.017 | 7/6 | 1.0  | [−.155, +.190] | NOT ESTABLISHED |

Read (registered interpretation; no verdict changes): at n=58 with
single-digit discordant counts, no answer-level margin in the
season reaches conventional significance. The machine-vs-RAG story
is directionally UNANIMOUS — the machine wins the discordant count
in all four pairings (6v1, 8v3, 8v6, 7v6), and the ordering arm
7v4 — but each margin individually sits within noise. C1's
non-inferiority likewise: the point estimate (−.017) sits at the
band's center, yet the CI cannot exclude losses beyond −.02.
Registered implication: the season's rankings are hypothesis-
generating, not confirmed at this n; the mechanical fix is more
frozen questions — a pre-registered question-set extension scoring
the SAME arms on fresh questions — recorded as a candidate next
series.

## D-series (registered 2026-10-15, pre-data): strengthening the verifier

The C-series closed with the binding constraint named: the top-8
pool holds gold .810 of the time, the verifier captures .466.
Disclosed pre-registration diagnostic (committed records only):
among pool questions where the scan REACHED gold and still missed
(11), gold's calibrated p sits below .20 in NINE — the judge
rejects gold confidently (median ≈ .03), and only one miss sits in
the rescueable .30–.50 band. The dominant failure is the reverse:
13 false-positive stops on plausible wrong pages. So sampling
variance is predicted NOT to rescue misses, and the tunable
quantity is the bar.

Questions: frozen 58. Corpus: pinned 5146e2d. Pool: committed
topk.json topk8.pick per question. Answer/grade instruments:
identical to A/B/C (GEN_SYS, grader verbatim, seed 20261014+17·i;
grader = generator, disclosed). Judge prompt: CR's JUDGE_SYS
verbatim in every instrument.

Instruments (both judged on ALL pool pages, full vectors):
- **S (single)**: one call, temp 0 — the committed instrument;
  live replication doubles as sanity (vs committed walkret p).
- **M (self-consistency)**: 3 calls, temp 1.0, seeds
  20261015+1013·i+17·s (s∈{0,1,2}); page score = mean of the 3
  raw p_yes, calibrated at committed T=0.66 (disclosed
  approximation: T was fit on single temp-0 calls).

Arms (all pool-confined, fallback rank-1, answers generated and
graded from the answered page):
- **D-M (primary)**: scan in pool order, stop at first page with
  cal(mean p) ≥ .5.
- **D-H**: scan the S vector, stop at first cal p ≥ .70 (higher
  bar vs false positives).
- **D-50**: scan the S vector at τ=.5 — no answers unless the
  answered page differs from committed ASM-8 (instrument
  replication sanity).
- **τ sweep (descriptive, chunk-level only)**: capture / fallback
  rate / would-be-fallback hit over τ ∈ {.30,.40,.50,.60,.70,.80}
  on both instruments' full vectors — post-hoc, like X's sweep;
  candidate τ values for a future split-tuned amendment.

**Bars (mechanical)**:
- **D1 (capture, D-M)**: answered-chunk hit ≥ committed .4655 +
  .05 = .5155 → PASS; ≥ .4855 → PARTIAL; else FAIL.
- **D2 (accuracy, D-M)**: ans_acc ≥ committed ASM-8 .4138 + .03 →
  PASS (verifier wins); ≥ −.02 → PARTIAL; else FAIL.
- **D3 (cost, D-M)**: total tok/q ≤ 5,427.7 (the FULL machine's
  budget — strengthened verification may not out-cost the scan it
  replaces) → PASS-budget; else FAIL-budget.
- **D4 (accuracy, D-H)**: ans_acc ≥ .4138 → PASS; ≥ −.02 →
  PARTIAL; else FAIL.
- Sanity (disclosed, not verdicts): live S-vector vs committed
  walkret p per-chunk stop-decision agreement ≥ .95 (expect
  1.000); D-50 answered-page agreement with committed ASM-8 ≥ .95.

**Registered bets (pre-data, from the disclosed diagnostic)**:
- **PD1: D1 expected FAIL** — the misses are confident rejections
  (9/11 below .20); 3-sample variance bridges ~1 question at most.
- **PD2: D2 expected PARTIAL** — any capture gain offset by more
  fallbacks (C-series: fallback answers grade .31 vs .49).
- **PD3: D3 expected PASS** (projected ≈ 4.9k tok/q: 3× sampling
  on ~3.7 pages + answers).
- **PD4: D4 expected FAIL** — the high bar converts false-positive
  stops into fallbacks, the weak link.
- Descriptive bets: the τ sweep shows capture roughly flat across
  .30–.60 (fp-stops and fallback gains trade ≈ evenly — the
  ceiling is the judge's discrimination, not the bar); M and S
  stop decisions agree on most questions; the capture gap
  (.810 pool vs captured) persists in every arm.

## G-series (registered 2026-10-15, pre-data): the grader audit

Every answer-level number this season rests on one grader that is
the same frozen model as the generator (disclosed since A-series)
with one prompt. This series stress-tests the instrument: re-grade
the committed answer STRINGS (no regeneration — answers are the
committed ones verbatim) under two altered grading instruments and
ask whether the season's ordering survives.

Scope: the four headline arms — ASM and TOP1 (assembly.json),
TOPK-2 (topk.json), ASM-8 (poolstop.json); n=58 committed answers
each. Grader model unchanged (the lab has one); only the grading
instrument varies.

Instruments:
- **G1 (reworded grader)** — system: "You are a careful grader.
  Reply with exactly one word: yes or no."; user: "Reference
  passage:\n{gold}\n\nQuestion: {q}\n\nProposed answer: {ans}\n\n
  Is the proposed answer a correct answer to the question,
  according to the reference passage?" — temp 0, logprobs, ok ≡
  p ≥ .5 (committed rule).
- **G2 (self-consistency grader)** — the ORIGINAL grading prompt
  verbatim, 3 samples at temp 1.0, seeds 20261016+101·i+13·s
  (s∈{0,1,2}); ok ≡ mean p ≥ .5.

Gates (mechanical):
- **GA (primary, per instrument)**: the season's ordering claim —
  ASM ≥ TOPK-2 AND ASM ≥ TOP1 — holds under the re-graded
  accuracies → PASS; any flip ≤ .02 → PARTIAL; any flip > .02 →
  FAIL.
- **GB (drift, per instrument)**: per-arm |re-graded acc −
  committed acc| ≤ .05 → PASS; else the drift is a disclosed
  caveat on that arm (and the audit says the instrument is
  prompt-sensitive).
Sanity: G1 on a 2-question smoke, marked, never scored.

**Registered bets**: GA PASS under both instruments (the ordering
is a real property of the answers, not the grading prompt); GB
PASS with G2 drifting less than G1; descriptive: G1 grades
slightly harsher (mean grade_p below the committed .43–.47 band).
If GA or GB fails, every answer-level number this season carries a
disclosed instrument-sensitivity caveat — that is the audit
working, not failing.

## D-series RESULTS (pooljudge.json, 2026-10-15 — scored)

Sanity: live S-vector stop decisions matched the committed walkret
p_yes on every judged chunk (agreement 1.000), and D-50 reproduced
the committed ASM-8 answered page on all 58 questions (agreement
1.000) — no answers needed, sixth consecutive session of temp-0
determinism.

| arm | answered hit | answer acc | direct | indirect | fallback | judged/q | tok/q |
|-----|--------------|-----------|--------|----------|----------|----------|-------|
| D-M (3-sample mean, τ=.5) | .466 | .414 | .621 | .207 | .276 | 1.74 | 2,827 |
| **D-H (single, τ=.70)**   | **.534** | **.448** | **.690** | .207 | .362 | 1.60 | **1,423** |
| (committed ASM-8, anchor) | .466 | .414 | .621 | .207 | .276 | 3.67 | 2,206 |

Verdicts, against the registered bets: **D1 FAIL** (Δ hit = .0000
vs the +.02 bar — PD1 landed: self-consistency rescued nothing),
**D2 PARTIAL** (Δ acc = .0000 — PD2 landed, though by identity,
not by offset), **D3 PASS-budget** (2,826.8 ≤ 5,427.7 — PD3
landed), **D4 PASS** (+.0345 — **PD4 refuted**, and in the useful
direction: the high bar is not a trade into the weak fallback
link, it is the best answer-level arm of the season).

The M-instrument null is total. The 3-sample mean produced the
SAME first-cross page as the single temp-0 call on 58/58 questions
— and the τ sweep rows for S and M are identical at every bar —
so D-M's answers came from the committed pages with the committed
seeds and reproduce ASM-8 number-for-number. The judge is
saturated: its misses are confident rejections (cal p < .20, as
the disclosed diagnostic showed) and its hits are confident
acceptances; three draws from that distribution average to the
same decision. Sampling variance is confirmed OUT as a rescue, at
3× the judge tokens (2,827 tok/q) for zero movement.

D-H's win, dissected. Raising the bar to .70 does two things at
once: it converts false-positive stops (13 at τ=.5) into
continuations that more often reach gold — first-cross capture
rises .379 → .397 — and it pushes more questions to the rank-1
fallback, whose quality RISES with the bar (fallback hit .086 →
.138 across the sweep; these 8 fallback-rank-1 golds are counted
in the .534 answered hit, the same convention as every arm this
season). Anatomy: stopped answers are right .622 of the time
(23/37), fallback answers .143 (3/21); direct recipe .690 (vs
.621 committed). Paired against committed ASM-8: D-H-only correct
2, ASM-8-only correct 0 — the entire +.0345. And it is cheap:
1,422.5 tok/q = 26% of the full machine, 64% of pool-confined
ASM-8.

The τ sweep (descriptive, chunk-level, as registered — no tuning
claim; D-H's .70 was fixed pre-data):

| τ | first-cross capture | fallback rate | fallback hit | hit + fb hit |
|------|------|------|------|------|
| .30 | .397 | .259 | .069 | .466 |
| .40 | .379 | .259 | .069 | .448 |
| .50 (committed) | .379 | .276 | .086 | .466 |
| .60 | .397 | .310 | .121 | .517 |
| **.70 (D-H)** | .397 | .362 | **.138** | **.534** |
| .80 | .276 | .500 | .224 | .500 |

Capture is flat .38–.40 from τ=.30 to .70 (the judge's
discrimination, not the bar, sets the ceiling — as the descriptive
bet said) and collapses at .80; fallback hit climbs monotonically
with the bar. The sweet spot is exactly where D-H was placed. A
split-tuned τ remains the registered future amendment if the
question set grows; on n=58 the honest summary is that the
committed .5 bar sits on a plateau and .70 sits on its edge.

Season consequence: the binding constraint named in C-series is
real but partially movable — not by strengthening the judge's
nerve (sampling does nothing) but by stiffening its spine (a
higher bar converts confident-wrong stops into mostly-recoverable
fallbacks). Best deployed arm is now .448 answers-correct at
1,423 tok/q. The capture gap (pool .810 vs hit .534) persists.

Tally: 23 of 41 headline predictions passed.

## G-series RESULTS (regrade.json, 2026-10-15 — scored)

Integrity: the re-grade used the same model that produced every
committed number — model fingerprint 7cfe33f65c1c identical across
assembly.json, topk.json, poolstop.json and regrade.json — and the
committed answer STRINGS were re-graded verbatim (no
regeneration). Committed anchors reproduced in-file: ASM .431,
TOP1 .345, TOPK-2 .397, ASM-8 .414.

| arm | committed | G1 (reworded) | G2 (3-sample) | G1 drift | G2 drift |
|-----|-----------|---------------|---------------|----------|----------|
| ASM    | .431 | .586 | .379 | +.155 | +.052 |
| TOP1   | .345 | .586 | .293 | **+.241** | +.052 |
| TOPK-2 | .397 | .466 | .379 | +.069 | +.017 |
| ASM-8  | .414 | .569 | .362 | +.155 | +.052 |

Verdicts: **GA PASS under both instruments** — the season's
ordering claim (ASM ≥ TOPK-2 AND ASM ≥ TOP1) survives every
re-grade (G1: margins +.121 / .000; G2: .000 / +.086). **GB FAIL
under both**: G1 drifts every arm past the .05 bar (TOP1 +.2414!);
G2 breaches it by .0017 on three arms (.0517). Registered bets:
GA both PASS — landed. GB PASS — refuted, both instruments. "G2
drifts less than G1" — landed (.052 max vs .241 max). "G1 grades
slightly harsher" — refuted in the opposite direction: G1 is
uniformly more LENIENT (mean grade_p .53–.54 vs the committed
.43–.44 band; every answer flip is upward, zero downward).

What the two instruments actually did. G2 (the faithful prompt,
sampled) is the committed grader with a slightly stricter edge:
3 answers per headline arm flip DOWN, none up, orderings intact —
the committed verdicts are robust to grader sampling noise. G1
(the rewrite) is a different instrument wearing the same name: it
rescues answers wholesale (+9 ASM, +14 TOP1, +5 TOPK-2; one
TOPK-2 answer flips down), and it rescues the weak arms MOST —
under G1, pure RAG's indirect answers invert the cell ordering
(TOP1 .517 vs ASM .379 indirect), the machine's overall margin
over TOP1 collapses to an exact tie, and only the direct cell
still separates them (.793 vs .655). The GA gate passes on a
technicality of the tie; the honest statement is that the
answer-level machine-vs-RAG margin is PROMPT-DEPENDENT — it is
+.086 under the committed prompt, .000 under the rewrite, +.086
under sampled-faithful.

Per the registration's own clause: every answer-level number this
season now carries a disclosed instrument-sensitivity caveat. The
caveat has a precise shape — chunk-level results (pool capture,
stop decisions, the τ sweep, token costs) are grader-independent
and untouched; answer-level ORDERINGS survived both audits
mechanically; answer-level MARGINS are soft at the ±.02–.09
scale, exactly the scale at which S-amendment already found every
margin within noise at n=58. The two independent stress tests
point at the same fix: more questions.

Tally: 25 of 45 headline predictions passed.

## F-series (registered 2026-10-15, pre-data): doc-selection

Every scored machine this season received the gold document for free —
A/B/C/D scanned chunks *within* the question's own doc; cross-document
ranking has never been measured. In deployment nothing marks the doc. This
series scores the layer above the pool: pick the document, then run the
scored machine inside it.

Scope: frozen 58, pinned corpus 5146e2d (4 docs / 180 chunks). Anchor:
committed D-H (gold-doc pool, τ=.70): acc .4483, hit .5345, 1,422.5 tok/q
(pooljudge.json). Instruments unchanged: extembed cache (plain query,
fp-verified), S verifier (single temp-0 judge call, cal T=0.66), grader =
generator, answers seeded 20261014+17·i, fallback rank-1.

**Doc-ranking rule (mechanical, registered)**: doc score = MAX over its
chunks of cosine(query, chunk); select argmax; ties broken by doc order.
(Disclosed: max-of-chunks, not mean-of-top-3 — one rule, no tuning.)

Arms:
- **F-DOC**: top-8 pool from the SELECTED doc (embedder order), scan at
  τ=.70, stop-and-answer, fallback pool-rank-1 — the committed machine one
  layer up.
- **F-GLOB**: top-8 pool across ALL 180 chunks (no doc layer at all), same
  scan. If this matches F-DOC, the doc layer is absorbed by global ranking.

Bars (mechanical):
- **PF1 (selection)**: doc-hit ≥ .95 → PASS; ≥ .90 → PARTIAL; else FAIL.
- **PF2 (F-DOC accuracy)**: acc ≥ .4483 − .02 → PASS; ≥ −.04 → PARTIAL;
  else FAIL.
- **PF3 (skip-the-layer)**: F-GLOB acc ≥ F-DOC − .02 → PASS; ≥ −.04 →
  PARTIAL; else FAIL.
- **PF4 (budget)**: F-DOC total tok/q ≤ 1,800 → PASS-budget; else
  FAIL-budget. (Doc ranking is cached-embedding lookups, ~0 marginal.)
- Sanity (not verdicts): pools on questions where the selected doc IS the
  gold doc must equal committed topk8.pick exactly; on those questions
  F-DOC answers must be byte-identical to committed D-H answers (temp-0
  determinism, same seeds); judge stop decisions vs committed walkret p_yes
  on shared pages ≥ .95 (expect 1.000).

**Registered bets**: PF1 PASS (the corpus's four docs are topically
distinct and the embedder's within-doc record is strong — but this is the
first cross-doc measurement, so the honest confidence is moderate); PF2
PASS; PF3 PARTIAL (global pools risk displacing gold with foreign-doc
neighbors in 1–3 questions); PF4 PASS; H-series-side: none.

## H-series (registered 2026-10-15, pre-data): abstention vs the fallback

The machine's registered fallback is rank-1 — answer anyway. A voice
assistant has a fourth option: say "I don't know." This series asks whether
a calibrated floor beats always-answering, using the committed D-H
instrument vectors ONLY — zero model calls (Z-series precedent for
committed-data-only amendments).

Scope: pooljudge.json records (58), D-H decisions at τ=.70 reproduced
mechanically from s_vec. **Floor rule (registered)**: on a fallback
question, abstain iff max calibrated p over the pool < φ; primary φ = .15 —
chosen a priori from the PUBLIC diagnostic (judged misses sit below .20
cal, median .03), before any per-question max-p was computed. φ sweep
{.05,.10,.15,.20,.30,.40} stays descriptive (coverage / precision /
acc-given-answered frontier) — no tuning claim, mirroring D's τ sweep.

Metrics (risk-relaxed, as posed in HANDOFF-wiresense.md):
- **acc-given-answered**: mean ans_ok over stopped ∪ kept-fallback
  questions (coverage reported alongside, always).
- **abstention precision**: fraction of abstained questions where the
  rank-1 answer was WRONG.

Bars (mechanical):
- **H1**: acc-given-answered(φ=.15) ≥ committed D-H .4483 (which counts
  fallback answers) → PASS; ≥ .4283 → PARTIAL; else FAIL.
- **H2**: abstention precision ≥ .70 → PASS; ≥ .50 → PARTIAL; else FAIL.
  Zero abstentions → FAIL (a floor that never fires is no instrument).
- **H3**: PASS-budget iff the runner makes zero LLM calls AND reproduces
  committed D-H stop/fallback decisions 58/58; else FAIL-budget.

**Registered bets**: H1 PASS (stopped answers .622 dominate the mean; the
floor only removes the weakest tail); H2 PASS (fallback answers are .143
correct overall — abstention is aiming at a mostly-wrong population);
H3 PASS. Disclosed risk: φ=.15 may abstain too little (if fallback
questions' max-p is inflated by a plausible-but-wrong top page) — the
descriptive sweep exists to map exactly that.

## F-series RESULTS (fselect.json, 2026-10-15 — scored)

Sanity: on correctly selected questions the doc pool matched the
committed topk8.pick exactly (6/6) and F-DOC reproduced the committed
D-H answer strings byte-for-byte (6/6) — seventh session of temp-0
determinism. Judge stop agreement vs committed walkret p = .976
(bar .95): the registered "expect 1.000" was wrong for a disclosed
reason — this comparison pits the τ=.70 bar against walkret's .50
threshold, so pages in the .50–.70 band count as disagreements; the
determinism claim rests on the answer-identity check, which is exact.

| arm | doc hit | answer acc | answered hit | fallback | judged/q | tok/q |
|-----|---------|-----------|--------------|----------|----------|-------|
| **F-DOC (select, then scan)** | .621 | .310 | .379 | .517 | 5.03 | 2,691 |
| F-GLOB (no doc layer, global top-8) | — | **.362** | .397 | .397 | 4.62 | 2,558 |
| (committed D-H, gold doc given) | 1.000 | .448 | .534 | .362 | 1.60 | 1,423 |

Verdicts: **PF1 FAIL** (.621 vs the .95 bar — the registered bet was
wrong, and it was the bet the registration itself flagged as
first-of-its-kind), **PF2 FAIL** (−.1379: mis-selection costs 14
points), **PF3 PASS** (+.0517: skipping the doc layer entirely BEATS
running it), **PF4 FAIL-budget** (2,691 tok/q vs the 1,800 ceiling —
mis-selected scans exhaust 8 judgments of the corpus's longest
chunks). One of four.

The mechanism is a chunk-count lottery, dissected. The four corpus
docs hold 9 / 32 / **137** / 2 chunks; the registered rule (doc score
= max chunk cosine) therefore draws 137 samples from the similarity
distribution for one doc and 2–32 for the others — the maximum of
more draws is larger by construction. The 137-chunk doc was selected
43/58 times overall and won 19 of the 22 mis-selections; the
32-chunk doc (24 gold questions) kept only 7. The indirect cell
selected its own doc half the time (.483) — exactly where the bet
risk was flagged.

F-GLOB's win is the mirror image: the global pool holds the gold page
only 31/58 (.534, vs .810 doc-confined — cross-doc displacement is
real, and 4 stops landed confidently in foreign docs), but GIVEN gold
in the pool the machine answers .645. The reader is unchanged; the
librarian's cross-doc list is the entire failure. The committed
season numbers (.448 @ 1,423) presuppose doc knowledge — worth
exactly .1379 of answer accuracy and half the budget on this corpus.

Registered consequence: the doc layer is neither free nor safely
skippable. Count-corrected selection, mean-of-top-3 doc scores, or
doc-centroid similarity are UNREGISTERED candidates — a future
amendment must commit to one before any scoring. Connection kept for
that amendment, not scored here: mis-selected scans are exactly where
the H-series floor abstains (their pools are confidently empty).

## H-series RESULTS (abstain.json, 2026-10-15 — scored)

Sanity: zero model calls by construction (H3's runner assertion) and
the committed D-H stop/fallback decisions reproduced 58/58 from the
committed s_vecs before any scoring.

Primary (φ=.15, chosen a priori from the public diagnostic): the
floor abstains on 11 of the 21 fallback questions (coverage .810),
and its aim is nearly perfect — **abstention precision .909** (10 of
the 11 withheld questions had a wrong rank-1 answer). The 10 kept
fallbacks answer .200; the 37 stopped answers stay .622; overall
**acc-given-answered = .532, +.0836 over the committed always-answer
machine (.448)**.

Verdicts: **H1 PASS** (+.0836 ≥ 0), **H2 PASS** (.909 ≥ .70), **H3
PASS-budget**. All three registered bets landed — the season's only
clean sweep where every bet was also correct.

The φ sweep (descriptive): acc-given-answered climbs monotonically
.464 → .481 → .532 → .533 → .558 across φ .05 → .30 while precision
stays in .83–1.00, and saturates at φ ≥ .30 (15 abstained, coverage
.741 — no fallback question's max-p lives in [.30,.40], consistent
with the saturated judge of D-series). The a priori φ=.15 sits at the
knee: nearly the precision of the strictest floor at twice the
coverage.

Season consequence: for a voice assistant, "I don't know" is now the
measured move — answer .532 when answering, withhold the 19% of
questions that are 91% wrong, and keep the rank-1 fallback for
everything above the floor. The abstention instrument costs zero
tokens: it is a threshold on confidence the machine already computes.

Tally: 29 of 52 headline predictions passed.
