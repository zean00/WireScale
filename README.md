# WireScale

*(formerly `tree-lab`)*

A pre-registered laboratory for decision structure. One frozen encoder, many
readouts — and one rule: **register the prediction before the data, then obey
the verdict.** Ten scored experiments live here:

1. **Decision trees** (P-series) — when an answer space is a hierarchy, is it
   cheaper to walk it level by level (asking a smaller question at each step)
   or to read it flat (choosing among all leaves at once)? Does telling each
   level what its parent answered help?
2. **Agent routing** (S-series) — a message must reach one of 64 specialist
   sub-agents behind 8 domains. What actually decides routing accuracy?
3. **Tool selection / filtering** (MS-series) — 121 real MCP tools on the
   wall, only ~65 useful per request. Run in the sister program
   [WireSense](https://github.com/zean00/WireSense); the complete
   experimental record — protocol (MS-1..MS-29), corpus, runners, scored
   results, run logs — is copied verbatim into this repo:
   [PROTOCOL-mcp-selection.md](PROTOCOL-mcp-selection.md) +
   `tools/mcp_probe|eval|loop/`.
4. **Chunk relevance** (R-series) — *scored 2026-10-13*: within-document
   chunk selection. Pointwise (question, chunk) joint judgment by the LLM
   vs bi-embedding cosine vs a trained pair head vs a listwise menu.
   **R1 PARTIAL** (the judge dominates cosine everywhere — but cosine
   itself sat at random-chance ranking, so the registered "repair" frame
   is confounded), **R2 FAIL** (the cacheable pair head keeps −67% of the
   gap), **R3 PASS** (the menu shortcut loses to exhaustive), **R4 PASS**
   (every confidence score is miscalibrated). Full record incl. one
   invalid run kept as disclosed record (corpus drift) in
   [protocol.md](protocol.md); runner `chunkrel.py`.
5. **Comparator autopsy** (V-series) — *scored 2026-10-14*: why was cosine
   dead in R — pooling, input format, or length? Five embedding arms
   (last/mean pool × with/without prefixes, lead-300) on the frozen 58.
   **VA PARTIAL** (mean-pool V2 gains +3.5 pts indirect — moves, but not
   rescuable), **VB pooling** (magnitude tie: pooling gains exactly what
   prefixes lose), **VC FAIL** (truncating chunks destroys direct recall).
6. **Retrieval walk vs flat** (W-series) — *scored 2026-10-14*: the founding
   "information beats structure" question re-asked where reads are
   expensive (exhaustive judging = 29k tokens/question). **W1 PASS**
   (flat doesn't lose), **W2 FAIL** (the 20-chunk-window walk collapses to
   0.17× flat recall despite costing 6% of the tokens), **W3 FAIL** (the
   walk is diluted inside the window, not just capped by its first step).
7. **Pair-head scaling** (J-series) — *scored 2026-10-14*: was R2's failure
   sample size or mechanism? 8× training pairs (230 → 1840). **JA FAIL**
   (val loss nearly halves, eval recall unchanged at floor — mechanism,
   not sample size), **JB PASS (degenerate)**.
8. **A real RAG embedder** (E-series) — *scored 2026-10-14*: swap the
   parasitic 4B comparator for a dedicated Qwen3-Embedding-0.6B (remote
   container) on the same frozen 58. **EA PASS** (+41.4 pts — the embedder
   beats even the exhaustive judge overall, .638 top-3 vs .517 yes-set
   recall, at ~zero marginal cost), **EB "instruct hurts"** (the model
   card's query format is a net negative here), **EC PASS** (5× random
   @3, 9× random @1). The lab's first decisive positive.
9. **Knowledge-graph QA** (K-series) — *scored 2026-10-14*: decision
   layers on graph traversal. A 902-triple graph extracted from the
   pinned corpus, 60 path-grounded audited questions, four arms.
   **K1 FAIL** (greedy hop-by-hop traversal loses to retrieve-then-read,
   −8.3 pts at 3.7× the tokens), **K2 PASS** (beam-2 beats the greedy
   walk +13.3 — its fourth consecutive replication and largest margin,
   keeping 89% of flat's accuracy at 6.5% of its tokens), **K3 PASS**
   (the flat anchor holds). *Amendment 2* (same day): 3-hop route
   lookahead — **KL1/KL2 FAIL** (route commitment collapses below
   edge hedging; the walker overshoots the answer, measured via the
   on-route diagnostic).
10. **End-to-end assembly** (A-series) — *scored 2026-10-14*: the recipe
   (embedder rank → scan-and-stop → calibrated bar → answer) scored as
   ONE machine on the frozen 58, answers generated and judge-graded
   end-to-end. **A1 PASS** (beats pure vector RAG +8.6 pts, .431 vs
   .345, at 18.6% of exhaustive-flat tokens), **A2 PASS** (embedder
   order beats doc order under the identical policy, +5.2), **A3 PASS**
   (answered-chunk hit .466 vs the committed .310 anchor — the
   registered fallback-to-top1 policy converts part of X's
   stop-vs-prefix gap), **A4 PASS-budget** (5.4k tok/q vs the 10.2k
   ceiling). Every registered prediction of the batch passed.
11. **The fair fight** (B-series) — *scored 2026-10-15*: A-series' RAG
   baseline was the strictest variant (top-1). Production stacks feed
   top-k excerpts; registered bets predicted top-k wins (context hit
   .638 at k=3 > machine's .466). **Both bars PASS anyway — the bets
   were wrong**: TOPK-3 scored exactly TOP1's .345 and the best
   affordable k (top-2, .397) still loses (B1 +.086, B2 +.034).
   Mechanism: synthesis given gold-in-context collapses .77 → .625 →
   .514 → .432 → .425 across k=1→8, cancelling the context-hit gain
   .448 → .810 — answer accuracy flatlines at .33–.40 across the whole
   sweep. The machine reads ONE judge-verified chunk and synthesizes
   .852. Disclosed: RAG keeps the per-token crown (top-2 holds 92% of
   the machine's accuracy at 21% of its cost).
12. **The pool-confined machine** (C-series) — *scored 2026-10-15*: the
   combination — RAG retrieves, the machine reads — registered with a
   disclosed post-hoc simulation as the pre-registered prediction, and
   the live run reproduced it exactly (stop-hit 27/58, fallback 16/58,
   sanity 1.000). Scan the top-8 pool with the calibrated judge, stop
   at the first verified page, answer from that page alone, fallback
   rank-1. **C1 PASS** (answer .414 vs the full machine's .431 — one
   question, where gold lives beyond the pool), **C2 PASS-budget**
   (2,206 tok/q = 40.6% of the machine's cost), **C3 PASS** (still
   ahead of the best RAG, .397). All registered bets landed.

**Every number in this repo is a committed artifact.** Hypotheses, arms, bars,
and amendments live in [protocol.md](protocol.md) — committed before any
scored run. Verdicts are computed mechanically ([adjudicate.py](adjudicate.py)),
not argued. A rendered, non-technical version of the whole story is at
[docs/index.html](docs/index.html) (GitHub Pages:
`https://zean00.github.io/WireScale/`).

## Headline results

| # | prediction (registered pre-data) | verdict |
|---|---|---|
| P1 | walking beats flat by ≥3 pts | **FAIL** — never, at any data size |
| P2 | parent-conditioning needs less data | **FAIL** — data buys back what structure promised (+2.4 pts at pool 16 → +0.7 at 1024) |
| P3 | conditioning helps most at low markedness | **FAIL as registered, substance confirmed** — at m=0 every arm sits at .517 ≈ coin flip; structure never beats information |
| P4 | the conditioning gap grows with depth (bar: +5) | **PARTIAL / null** — +0.5 pts; stopped by the registered rule |
| R1 | LLM judge repairs indirect retrieval (gap ≥ +10 pts) | **PARTIAL** — gap +10.3, but cosine sat at random at both recipes: dominance, not repair |
| R2 | trained pair head keeps ≥50% of the gap | **FAIL** — keeps −67% (below the comparator, below random) |
| R3 | the menu shortcut does not beat exhaustive reading | **PASS** — 0.33 vs 0.52 |
| R4 | some arm's confidence is miscalibrated (ECE ≥ .05) | **PASS** — ECE 0.10–0.14, both probabilistic arms |
| VA | some pooling/format/length dial rescues the comparator (indirect gap ≥ +10) | **PARTIAL** — mean-pooling +3.5 pts (top-1 ×2.2); no dial rescues it |
| VC | length fix (lead-300) gains ≥ +5 direct | **FAIL** — −6.9 pts |
| W1 | flat does not lose to the walk | **PASS** — +43.1 pts |
| W2 | walk keeps ≥80% of flat recall at ≤35% tokens | **FAIL** — 17% of recall at 5.8% of tokens |
| W3 | the walk is capped by its first step, not diluted inside | **FAIL** — diluted inside too (.294 given the right window) |
| JA | the pair head learns with 8× data (gap ≥ +5 over cosine) | **FAIL** — −10.3 pts; val loss halves, eval unchanged at floor |
| EA | a dedicated embedder rescues the comparator (indirect gap ≥ +10) | **PASS** — +41.4 pts; beats the judge overall (.638 vs .517) |
| EC | the dedicated embedder clears 2× random @3 | **PASS** — .638 ≈ 5× random; top-1 9× random |
| K1 | traversal beats retrieve-then-read at matched budget | **FAIL** — −8.3 pts at 3.7× the tokens |
| K2 | beam-2 beats the greedy walk by ≥ +3 on graphs | **PASS** — +13.3, fourth replication, largest margin |
| K3 | flat anchor: whole-graph read not beaten by a budgeted arm | **PASS** — +3.3 over beam |
| KL1 | 3-hop route lookahead beats edge-hedging beam | **FAIL** — −25 pts; collapses to 2/60 |
| KL2 | route lookahead rescues the greedy walk | **FAIL** — −5 pts (deeper commitment, not less) |
| XA | embedder-ordered early-stop keeps ≥95% of flat recall at ≤25% tokens | **PASS** — .690 prefix (> flat's .517) at 15.8% of tokens |
| XB | embedder order beats doc order at deployment (stop-recall) | **PASS** — .397 vs .310 |
| YA | rewriting the query helps indirect retrieval (≥ +5) | **FAIL** — −6.9 pts; the raw question wins |
| YB | rewriting doesn't damage direct retrieval (≥ −3) | **FAIL** — −3.45 pts |
| KM1 | under mid-route scoring, beam traversal beats retrieve-then-read | **PASS** — .300 vs .233 |
| ZA | temperature calibration improves held-out ECE by ≥ .01 | **PASS** — .105 → .081 (T = 0.66) |
| ZB | the calibration is stable across the split | **PASS** — direction agrees; stop-bar τ transfers identically |
| A1 | the assembled machine beats pure vector RAG by ≥ +3 pts | **PASS** — .431 vs .345 (+8.6) at 18.6% of flat's tokens |
| A2 | embedder ordering beats doc order at answer level (same policy) | **PASS** — .431 vs .379 (+5.2) |
| A3 | fallback policy: answered-chunk hit ≥ committed X-DS .310 | **PASS** — .466 (+15.5 over the anchor) |
| A4 | the machine fits the budget (≤ 10,233 tok/q) | **PASS-budget** — 5,428 tok/q |

Depth extension to 6 levels / 96 leaves: **no depth wall** (levels 5–6 are
free, .9997 and 1.0000), but the walk is capped by its first step (.918 at
L1 → .825 overall), **flat overtakes the walk at 96 classes** (.899 vs .825),
**beam-2** (keep two first-level candidates) recovers +3.4 pts, and the FiLM
conditioning layer was reliably **harmful** (−13.4 pts) — rejected with
numbers, not taste.

Agent routing (S-series, 8 domains × 8 agents = 64): **nameability is the
whole game.** S1 PASS — beam-2 beats the greedy walk +7.7/+3.3 pts (its third
consecutive replication). S2 — the flat 64-way glance beats the walk at every
nameability level (.929 vs .796 at m=1). S3 — at m=0 every arm lands at the
1/8 floor, *including the cheating oracle*, while departments stay at .8555
(registered bar .90, missed exactly as predicted pre-data). Pass 1 of the
S-series was invalid (scoring bug + substrate flaw) — deleted unexamined and
disclosed in protocol.md.

Chunk relevance (R-series, 4 docs / 180 chunks / 58 frozen eval questions):
**the judge dominates, the comparator collapses.** Bi-embedding cosine
(slot 7) ranks question→chunk at random within-doc at BOTH markedness
levels; the exhaustive LLM judge collects the gold chunk 0.79 (direct) /
0.24 (indirect) vs cosine 0.21 / 0.14 — and gains land exactly where
cosine dies (the 137-chunk doc: 0.63 vs 0.00). The trained pair head
failed outright (worse than cosine), the 20-preview menu loses to
exhaustive reading at 24× fewer tokens, and every confidence score is
miscalibrated (ECE 0.10–0.14). One invalid run kept as record: the
operator's edit to protocol.md (a corpus doc) shifted chunking under the
frozen questions — caught post-run, disclosed, corpus pinned, rerun.

The three follow-ups (2026-10-14) closed every escape hatch. **V-series:**
no pooling/format/length dial rescues bi-embedding on this encoder —
mean-pooling is the only dial that moves it (+3.5 pts indirect, top-1
doubles to .155), prefixes actively hurt, truncation destroys direct
recall; even the best configuration (.19) barely clears the @3 random
baseline (.128). **W-series:** where reads are expensive, structure
still loses — the 20-chunk-window walk keeps 17% of flat recall at 6%
of the tokens; the window selector (crude previews) picks the right
window 43% of the time, and even with gold in scope the judge fires at
half its flat rate. The post-hoc sweep found the actual economics win:
an *early-stop rule over the exhaustive scan* (stop when p_yes crosses
.5) matches flat recall while judging ~1/10 of the chunks — post-hoc,
must be re-registered before use. **J-series:** R2's failure was
mechanism, not sample size — 8× training pairs halve validation loss
while eval recall stays exactly at floor; the [q;c;|q−c|;q·c] features
cannot separate gold from same-doc distractors at any tested scale.

Then the last escape hatch gave way (E-series, 2026-10-14): swap the
parasitic encoder for a **dedicated Qwen3-Embedding-0.6B** — the kind
of model real RAG stacks use — and the whole picture flips. Top-3
.638 overall vs the judge's .517 yes-set recall (a looser criterion),
indirect .552 vs the judge's .241, top-1 9× random, the 137-chunk doc
solved (.542 where the parasitic encoder scored .000). Cost per query:
one cached-embedding lookup vs 29k tokens / 19 s of judging. Every
prior "the comparator is dead" verdict was a fact about the 4B model
doing embeddings as a side job — dense retrieval itself was never on
trial. The judge keeps one crown: direct-question yes-set recall
(.793 vs .724). And one anti-lesson: the model card's prescribed
instruct-wrapped query format *hurt* here (−3.5 pts) — measured, not
assumed.

Then the graph question (K-series, 2026-10-14): a 902-triple knowledge graph
extracted from the pinned corpus, 60 audited path-grounded questions, and
decision-layer traversal put to its first fair test with a good cheap
selector. **Greedy traversal lost again** (K1 FAIL: −8.3 pts to
retrieve-then-read, at 3.7× the token budget) — the sixth consecutive
verdict that committing to one path early is the wrong shape. **But beam-2
replicated a fourth time and by its largest margin** (+13.3 over the greedy
walk), landing within 3.3 pts of reading the whole graph flat while spending
6.5% of its tokens. The one structural pattern this lab has ever found
paying — keep two candidates alive instead of one — now holds on trees,
routing, and graphs. (One incident kept as record: the first scored attempt
crashed on question 1 — the whole-graph prompt exceeded the production
slot's 8192-token ceiling — amended pre-data to FLAT-2HALF, a deterministic
two-half read.)

The lookahead follow-up (K-series Amendment 2, 2026-10-14): the operator
proposed fixing the greedy walk's myopia by choosing a whole 3-hop route at
a time instead of one edge. Registered pre-data, scored against the
committed baselines. **Both verdicts FAIL, decisively** — route-level
commitment is still commitment: LOOK-1 (single route, the proposal as
stated) landed *below* the plain greedy walk (−5 pts), and LOOK-2 (frontier
of 2 routes) collapsed to 2/60, −25 pts vs beam. The diagnostic explains
it: the gold node sat on the chosen route 32% of the time, and in
practically every one of those cases the walker walked PAST it — a 3-hop
route overshoots a 2-hop answer. The design instinct was half right (myopia
was the correct diagnosis); the fix that survives measurement is wider
hedging, not longer committed routes.

The follow-up batch (X/Y/K3/Z, 2026-10-14) turned the diagnostics into the
production recipe. **X-series (early-stop judging on embedder order) — both
PASS**, and the batch's headline: scanning chunks in embedder rank order
and stopping at the first confident yes reaches .690 prefix-recall —
*above* the exhaustive judge's .517 — while reading a mean 11.5 of 180
chunks at 15.8% of the tokens; judge decisions reproduced committed scores
with agreement 1.000. (Registration also corrected the record: W-series'
"early-stop" post-hoc was order-free — the sequential scan had never
actually been run until now.) **Y-series (query rewriting) — both FAIL**:
the rewriter genuinely rewrote (96.6% changed) and retrieval got worse on
both cells — second anti-lesson about "helping" the query, after instruct
prefixes. **K-series Amendment 3 (mid-route scoring) — KM1 PASS**: when
walkers get credit for any node they visit, beam beats retrieve-then-read
(.300 vs .233), and the LOOKAHEAD twist lands — its routes pass gold a
third of the time, so its mid-score (.333) is the best graph number in the
lab, validating the overshoot diagnosis. Endpoint drift of the fresh run:
exactly 0.000. **Z-series (calibration) — both PASS**: one temperature
(T = 0.66 — the judge is overconfident) cuts held-out ECE .105 → .081, and
the 90%-of-flat stop-bar (τ = .05) transfers identically across the split.

The assembly (A-series, 2026-10-14) closed the loop: the recipe scored as
one machine on the frozen 58, answers generated and judge-graded
end-to-end. **All four PASS.** The machine answers .431 vs pure vector
RAG's .345 (A1) and vs the same policy run in document order .379 (A2) —
every component's contribution survives composition. The registered
fallback (nothing crosses the bar → take the embedder's top pick; it
fired on 21% of questions) lifts answered-chunk hit to .466 vs the
committed .310 anchor (A3), cashing in part of X's stop-vs-prefix gap.
Cost: 5.4k tok/q, 18.6% of the exhaustive scan (A4). Remaining losses,
disclosed in the RESULTS: answer synthesis drops ~3.4 pts even with the
right chunk in hand, and the indirect cell is still the frontier (.207).
The fair fight (B-series, 2026-10-15) then held the machine against
production-style top-k RAG — the registered bets said the machine loses,
the bars said otherwise: multi-excerpt RAG flatlines at .33–.40 no matter
how many pages you stuff in (dilution eats the recall gain), while
verified-single-chunk synthesis holds .852.

**21 of 37 headline predictions passed as registered.** The failures
keep mapping the dead ends (rewriting, route commitment, pooling dials,
trained heads — and the operator's own bet that plain top-k RAG would
beat the machine); the passes now form a recipe — and the recipe is now
a scored machine that survived its fair fight, then got 2.5× cheaper by
letting RAG draw its reading list: **embed with the dedicated model,
scan its top-8 pool, stop when the calibrated judge is confident, fall
back to the rank-1 pick when nothing crosses, hedge on graphs, and
read flat when flat fits.**

What transferred to production: beam-2 path scoring in the WireSense router —
which only worked after discovering paths of different depth must be scored by
**mean, never sum** (a 13-point crash hid in that choice) — and the whole
tool-selection program it seeded (filter 121 tools to ~65 per request;
current wiring target 73.4% vs 67.3% for the shipped gate).

## Layout

| path | what |
|---|---|
| `protocol.md` | the decision record: questions, arms, registered predictions, amendments, results — the source of every number above |
| `hierarchy.py` | synthetic utterance-marked hierarchy (deterministic, seed 0); the markedness dial m |
| `heads.py` | trunk + per-level classifiers, arms A/B/C/C′ (C = parent one-hot concat, C′ = FiLM) |
| `walks.py` | hard walk, beam-2, flat; oracle-parent teacher forcing; metrics |
| `run_lab.py` | registered P-series sweeps → append-only JSON in `results/` |
| `subagents.py` | S-series runner (raw-embedding centroids, no trained heads) |
| `chunkrel.py` | R-series runner (chunk relevance: bi-embed cosine V vs pointwise judge J-llm vs pair head J-combo vs listwise J-menu) — **scored 2026-10-13 (R1 PARTIAL, R2 FAIL, R3/R4 PASS), corpus pinned to the freeze-audit commit** |
| `vautopsy.py` | V-series runner (pooling/format/length autopsy of the cosine comparator) — **scored 2026-10-14 (VA PARTIAL, VB pooling, VC FAIL)** |
| `walkret.py` | W-series runner (window walk vs exhaustive judge) — **scored 2026-10-14 (W1 PASS, W2/W3 FAIL)** |
| `pairscale.py` | J-series runner (8× pair-head scaling; `--gen` writes the unaudited train set) — **scored 2026-10-14 (JA FAIL, JB degenerate PASS)** |
| `extembed.py` | E-series runner (dedicated Qwen3-Embedding-0.6B via SSH tunnel; own cache) — **scored 2026-10-14 (EA/EC PASS, EB instruct hurts)** |
| `kgraphqa.py` | K-series runner (graph build from pinned corpus, path-grounded questions, FLAT/READ/WALK/BEAM) — **scored 2026-10-14 (K1 FAIL, K2/K3 PASS)** |
| `results/kgraph.json`, `results/k_questions.json` | the K-series substrate: 902-triple extracted graph + the audited frozen 60 |
| `kgraphqa.py --look`, `klookahead.json` | K-Amd 2 (route lookahead) — **scored 2026-10-14 (KL1/KL2 FAIL)** |
| `kmid.py`, `kmid.json` | K-Amd 3 (mid-route rescoring) — **scored 2026-10-14 (KM1 PASS)** |
| `xstop.py`, `xstop.json` | X-series (sequential early-stop scan) — **scored 2026-10-14 (XA/XB PASS)** |
| `qrewrite.py`, `qrewrite.json` | Y-series (query rewriting) — **scored 2026-10-14 (YA/YB FAIL)** |
| `calib.py`, `calib.json` | Z-series (temperature calibration, committed data only) — **scored 2026-10-14 (ZA/ZB PASS)** |
| `asm.py`, `assembly.json` | A-series (end-to-end assembly: TOP1 vs FLATPOL vs ASM, fallback-to-top1, judge-graded answers) — **scored 2026-10-14 (A1–A4 all PASS)** |
| `topk.py`, `topk.json` | B-series (the fair fight: TOPK-k k∈{2,3,5,8} vs committed ASM, budget-dominance bars) — **scored 2026-10-15 (B1/B2 PASS; registered bets wrong)** |
| `poolstop.py`, `poolstop.json` | C-series (pool-confined machine: scan the top-8 pool, stop-and-verify, fallback rank-1; ASM-8 + ASM-4) — **scored 2026-10-15 (C1–C3 all PASS; simulation reproduced exactly)** |
| `adjudicate.py` | mechanical P1–P4 verdicts against the registered bars |
| `posthoc_p3.py` | the post-hoc m=0 probe behind P3's verdict |
| `embed_client.py` | embedding client for the frozen encoder |
| `results/` | scored runs, append-only; `subagents_pass1_invalid.json` ships as the disclosed record of the deleted run |
| `models/`, `tools/decision_tree.json` | the production CLM readout stack (head trunk, per-node reference banks, tree spec) — copied from WireSense, **model-bound**, see `models/README.md` |
| `PROTOCOL-mcp-selection.md` | the MS-series decision record (tool selection, MS-1..MS-29) — copied verbatim from WireSense |
| `tools/mcp_probe/` | MS-series runners, MCP corpus (10 real servers, dumped verbatim), eval cases, synth pools, scored results + run logs — copied verbatim |
| `tools/mcp_eval/`, `tools/mcp_loop/` | per-server implementation evals (MS-16 era) and mid-loop gate probes (MS-17a/b) — copied verbatim |
| `tools/mcp_gate.py`, `tools/mcp_client.py` | the shipped production coarse gate + MCP transports (WireSense `web/`) — use case 3's production artifact, copied verbatim |
| `docs/index.html` | the GitHub Pages page (single file, no dependencies) |

## Running it

Everything is model-bound to a frozen encoder: Qwen3.5-4B served by a local
llama.cpp server with `--embedding --pooling last` (2560-dim). The lab reads
embeddings from a **dedicated slot (id_slot 7)** — the production readout slot
is never touched by lab runs.

```bash
python -m venv .venv && .venv/bin/pip install numpy torch   # already present here
.venv/bin/python run_lab.py --smoke     # plumbing check (marked smoke, never scored)
.venv/bin/python subagents.py --smoke   # tiny routing check
.venv/bin/python chunkrel.py --smoke    # R-series plumbing check (never scored)
.venv/bin/python vautopsy.py --smoke    # V-series plumbing check (never scored)
.venv/bin/python walkret.py --smoke     # W-series plumbing check (never scored)
.venv/bin/python pairscale.py --smoke   # J-series plumbing check (never scored)
.venv/bin/python extembed.py --smoke    # E-series plumbing check (never scored)
.venv/bin/python asm.py --smoke         # A-series plumbing check (never scored)
.venv/bin/python topk.py --smoke        # B-series plumbing check (never scored)
.venv/bin/python poolstop.py --smoke    # C-series plumbing check (never scored)
.venv/bin/python adjudicate.py          # recompute P1–P4 verdicts from results/
```

V-series' mean-pool arms need a second, CPU-only llama-server on
127.0.0.1:8999 (`--pooling mean`, same model file, GPU hidden via
`CUDA_VISIBLE_DEVICES=`); without it the runner reports the mean arms
unavailable. E-series needs the remote embedding container (operator's
host) plus an SSH tunnel: `ssh -N -L 8124:127.0.0.1:8124 <host>`.

All scored runs are already recorded and committed under `results/` — reruns
append, they never overwrite. Changing the encoder, quantization, or pooling
invalidates every embedding-derived artifact in the repo (see the disclaimer
in `models/README.md`).

The MS-series files (`PROTOCOL-mcp-selection.md`, `tools/mcp_*`,
`tools/mcp_gate.py`, `tools/mcp_client.py`) are a **verbatim copy** of the
sister repo's tool-filtering experiment (WireSense @ `876e1e8`; diff-clean,
including the four `.log` files, which the root `.gitignore` exempts
explicitly). Its runners `import server` — the production planner at
`web/server.py` relative to WireSense's root — so in this copy they are the
**frozen record** of scored runs: re-running them requires a WireSense
checkout; the committed JSONLs, summaries, and run logs are the results.
Final MS-series state: MS-29 `winhint` (window-union shortlist +
intent-gated hint) is the registered production wiring target — 0.7338 vs
0.6727 for the shipped gate, 0.6835 for flat — with production integration
still pending in WireSense.

---

Sister program: [WireSense](https://github.com/zean00/WireSense) — the voice
assistant where the production transfers live. Experiments ran 2026-10-07 →
2026-10-09. Register first, measure, admit when wrong.
