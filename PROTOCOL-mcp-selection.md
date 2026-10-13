# PROTOCOL: MCP tool-selection experiment (flat vs description-built hierarchy)

Status: REGISTERED 2026-10-06, pre-data. Question: before implementing MCP support,
does hierarchical tool selection built from tool descriptions beat a flat full-manifest
planner on accuracy, and at what token/latency cost — on the production 4B planner.

## Motivation

Stage 3 measured that growing the manifest 4→6 tools caused a real bleed
(weather → ir_research, caught by battery, fixed by Rev G plan-twice). MCP support
would mean 30–100+ tools from third-party servers. This experiment measures the
selection surface BEFORE any integration code exists. n_ctx = 8192 (measured via
/props at registration) means a 100-tool flat manifest cannot carry full
descriptions; the truncation budget is part of the registered design, not an
improvisation.

## Corpus (MS-1)

Real public MCP servers, runnable locally without credentials, executed once via a
minimal stdio MCP client (initialize → tools/list), their tool JSON dumped VERBATIM
to `tools/mcp_probe/corpus/<server>.json`. No hand-edited descriptions, no synthetic
tools. Target: ≥100 tools across ≥8 servers. Near-duplicate-rich servers (github,
git) included DELIBERATELY — within-domain confusion is a claim under test.

Registered server list (with named alternates if one fails to start):

| server | runner | note |
|---|---|---|
| filesystem | npx @modelcontextprotocol/server-filesystem | official |
| memory | npx @modelcontextprotocol/server-memory | official |
| sequential-thinking | npx @modelcontextprotocol/server-sequential-thinking | official |
| puppeteer | npx @modelcontextprotocol/server-puppeteer | official |
| everything | npx @modelcontextprotocol/server-everything | official test server |
| git | uvx mcp-server-git | python ref |
| time | uvx mcp-server-time --local-timezone Asia/Jakarta | python ref |
| fetch | uvx mcp-server-fetch | python ref |
| sqlite | uvx mcp-server-sqlite --db-path (empty temp db) | python ref |
| github | github/github-mcp-server release binary, stdio mode | ~40 tools, Go |

Alternates (only if a listed server cannot start): mcp-server-postgres (temp db),
archived @modelcontextprotocol/server-github (dummy token). Skipped for credentials:
slack, gdrive, brave-search. A failed server is recorded with its error; the corpus
commit records final membership and per-server tool counts.

## Arms (MS-5)

All arms share: production `build_prompt(history=[], utterance, directive)` with the
manifest swapped (MCP tools instead of TOOL_MANIFEST), production `parse_tool_json`,
temperature 0, id_slot 5, max_tokens 120 (production uses 80; raised only because
MCP arg schemas demand more required-arg emission — tool choice is unaffected),
history-free planning (Rev G rationale: the job must stand on its own words).

| arm | routing | planner sees |
|---|---|---|
| flat | none | all N tools |
| hier-desc/top1 | centroid route, argmax cluster | top-1 cluster's tools |
| hier-desc/top2 | centroid route, top-2 clusters | union of top-2 clusters' tools |
| hier-synth/top1 | synth-utterance centroid route | top-1 cluster's tools |
| hier-synth/top2 | synth-utterance centroid route | union of top-2 clusters' tools |

Every arm's directive ends with the production JSON contract, including
`{"tool": null}` escape. Hier routing embeds the utterance once (norm_router_text →
POST /v1/embeddings id_slot 6, production path) — cluster readouts are numpy matmuls.

## Context budget (MS-2)

Planner-visible per-tool fields: name + description truncated to C chars + names of
required args from inputSchema. C chosen once at runner freeze from {200, 160, 120,
100, 80} as the LARGEST value whose measured flat-arm prompt_tokens ≤ 7168 (8192
minus headroom for template + utterance + 120 output). Measured on one canary case,
recorded in the results commit BEFORE any scored run. The same per-tool truncation
applies inside hier shortlists. Embedding pools always use the FULL description
(that is the hierarchy's selling point: full info at routing, shortlist at planning).

## Routing pools (MS-3)

- hier-desc: one vector per tool = L2-normalized embedding of `"{name}: {description}"`.
- hier-synth: per-tool pool of 8 synthetic user utterances (LLM slot 5, temp 0.8,
  seed 1, recipe A: typical direct request phrasing, ≤12 words, no tool name),
  centroid = mean of L2-normalized embeddings. KNOWN BIAS, disclosed: eval
  direct-slice shares this register by construction; counterweights are the
  indirect eval slice, hand-written traps, and hand-written negatives.
- Clustering: k-means, k=10, cosine space, greedy farthest-point init seed 0,
  50 iterations, numpy only (deterministic; k fixed at corpus freeze from
  k ≈ sqrt(N_tools)).

## Eval set (MS-4)

Per tool, 2 positives (slot-5 generated, temp 0.8, seed 2, recipe B; ground truth =
tool name): B-direct (direct request; tool name not copied verbatim where unnatural)
and B-indirect (goal-oriented, tool unnamed). Plus hand-written: 24 negatives
(production-style no-tool utterances: chit-chat, math, and the domains wiresense
already owns — timer/reminder/note/detect/weather) expecting `{"tool": null}`;
12 traps (cross-server confusable but decidable). ALL cases hand-audited for
ground-truth validity BEFORE any scored run; corrections only pre-freeze; the
freeze commit is immutable thereafter. ~236 cases total.

## Metrics (MS-6)

Primary: overall accuracy = (positives with exact tool match + negatives answered
null) / all cases. Secondary: positive accuracy, negative accuracy (null precision),
args-schema-validity rate among correct picks (required fields present and typed),
mean prompt_tokens, mean wall-clock ms per decision, hier cluster-assignment accuracy.

## Hypotheses and verdict rule (MS-7)

- H1: hier-synth/top2 overall accuracy ≥ flat.
- H2: flat mean prompt_tokens ≥ 5× hier mean prompt_tokens.
- H3: hier-synth routes better than hier-desc (cluster-assignment accuracy).
- H4: the accuracy gap grows with tool count (gradient arm: flat vs best hier on
  seed-0 server subsets at ~25/50/100 tools; runs after the main verdict, same rules).

Adoption rule: recommend building MCP selection hierarchically iff H1 holds AND
hier negative-accuracy ≥ flat negative-accuracy − 2 points (hierarchy must not
buy accuracy by bleeding more on no-tool cases). Otherwise recommend flat with
truncation, and record the measured token cost.

## Adjudication clauses (MS-8)

- Single-case temp-0 serving variance: one identical-prompt rerun permitted, both
  results recorded, rerun binding (stage-3 precedent).
- Infra failure (server down, spawn fail, timeout): case rerun, logged.
- Corpus and eval are immutable after their freeze commits; rule changes require a
  new numbered decision committed BEFORE the affected data exists.
- Gate data = scored eval results. Corpus counts and canary timings are not gate data.

## Out of scope (MS-9)

Live server lifecycle (connect/disconnect/rebuild-on-change), authenticated servers,
args QUALITY (only schema-validity is scored), production integration point
(think.tool subtree vs new root class), multi-tool chains. These are
implementation-stage decisions, taken after this read.

## Runner freeze (MS-10)

Runners committed before scored runs; 6-case canary green; then one full pass over
all arms in a single session; results appended to this document with a VERDICT line;
failures recorded as failures.

## Results

(appended post-run)

## Amendment MS-11 (registered post-run-1, pre-any-hier-rerun-data): routing union bug

Run 1 (2026-10-06): canary green mechanically (budget C=200, flat prompt 3738
tokens; parse/scoring/resume all exercised), full 5-arm pass completed — but the
hier arms are INVALID. run_eval.py's shortlist union indexed the per-tool cluster
assignment with centroid indices (`np.argsort(-sims)` iterates k centroids, used
as tool indices), so hier shortlists were 2–4 arbitrary tools from the corpus head
(mean hier prompts 130–196 tokens vs an expected ~700–2500). Diagnosed from the
route records: positives answered null at ~80%, pos_acc 3.5–6.7% ≈ random over
tiny wrong shortlists. Flat arm is unaffected by the bug (no routing) and is
CARRIED OVER byte-identical from run 1 (278 records). Decision: fix the union to
`members = [i for i, a in enumerate(assign) if a in top_clusters]`; archive run-1
hier records verbatim as results_run1_invalid_hier.jsonl (committed); rerun the
four hier arms in full under the same rules. No arm definition, eval case, budget,
or metric changes.

## Amendment MS-12 (registered post-run-2, pre-sweep-data): granularity sweep + scale gradient

Run 2 decomposition (analysis, not gate data): hier loss vs flat is entirely
routing recall — hit rate 43.3–71.3% across hier arms (was the right tool in the
shortlist at all), while pos_acc|hit is 75–91%, ABOVE flat's 66.5% overall
(shortlists de-crowd the planner). pos_acc|miss = 0.000 in all arms. Cluster
granularity (k=10, fixed by MS-3 heuristic) is therefore the load-bearing design
parameter, and the verdict "hierarchy loses" could be an artifact of k.

Registered follow-ups, both pre-data:
1. k-sweep: hier-synth/top2 (best arm by overall accuracy) re-run with
   k in {5, 8, 15, 25}, everything else identical. Runner gains env MCP_PROBE_K
   (clustering k only). Question: does any k beat flat's 68.4% overall?
   Note the monotone intuition this tests: smaller k means higher recall but
   weaker de-crowding; k=1 degenerates toward flat-with-truncated-descriptions.
2. H4 gradient (registered in MS-7, now runnable): flat vs hier-synth/top2 at
   the best k from (1), on seed-0 SERVER subsets (np.random.RandomState(0)
   shuffle of server names, cumulative prefix >= 25 and >= 50 tools; k for a
   subset = round(sqrt(N_subset)), consistent with MS-3's rule). Cases whose
   want-tool is not in the subset are DROPPED for positives/traps; all 24
   negatives stay (bleed test is valid — and relatively harder — at every
   size). Runner gains env MCP_PROBE_SUBSET=<N>. Question: does the
   flat-vs-hier gap shrink or invert as the manifest shrinks?

## Amendment MS-13 (registered post-invalid-sweep, pre-sweep-rerun): dead-constant bug

MS-12's first sweep (k=5, 8, 15 complete; 25 partial) is INVALID: a leftover
`K = 10` constant three lines below the new env read overwrote MCP_PROBE_K at
import, so every sweep arm silently ran k=10 — caught because all files were
byte-identical in accuracy, token count AND route clusters (ids reaching 9 under
a nominal k=5). Sweep files kept on disk outside the repo as run-1-sweep;
invalid; nothing from them enters the verdict. Fix: delete the dead constant
(single line), verify with a 40-vector probe (K=5 yields 5 centroids), rerun the
full sweep per MS-12's text. Killing the in-flight background run counted as
the infra-failure stop of MS-8.

## Amendment MS-14 (registered post-gradient-flat, pre-hier-gradient): subset degeneracy + path bug

Two findings from the gradient launch:
1. Registered 25/50-tool server subsets are UNREACHABLE in this corpus: the
   seed-0 server order starts with github (62 tools), so the smallest whole-
   server prefix already exceeds both targets. Both "subsets" resolved to the
   same 78-tool prefix (github, filesystem, fetch, sequential-thinking; 186
   cases). Decision: the gradient is answered at N=78 and labeled as such; a
   true 25-tool point would need tool-level subsetting (unregistered — not
   done). Free bonus from the duplicate: two independent temp-0 flat runs on
   identical prompts produced 0/186 differing verdicts — the pipeline is
   deterministic at same-prompt temp 0, so arm differences on this eval are
   configuration properties, not run noise.
2. The two hier gradient runs crashed BEFORE any scored case: the arm name
   contains a slash and was interpolated into the results path, pointing at a
   nonexistent directory (infra-failure class per MS-8; flat runs completed).
   Rerun: hier-synth/top2 k=5 at the N=78 subset only, path sanitized.
Measured already (flat@78): acc 0.3602, pos 0.3765, neg 6/24, tokens 2870 —
a one-server-dominated flat manifest collapses selection AND the null escape;
recorded as a main result, not an outlier, pending the hier comparison.

## Results (run 2 valid; adjudicated 2026-10-06)

Main arms, 278 cases (242 positives + 24 negatives + 12 traps), 121 tools, k=10:

| arm | acc | pos_acc | neg_acc | schema_ok | prompt_tok | ms |
|---|---|---|---|---|---|---|
| flat | 0.6835 | 0.6654 | 21/24 | 0.953 | 3739 | 2205 |
| hier-desc/top1 | 0.4317 | 0.3937 | 20/24 | 0.990 | 680 | 668 |
| hier-desc/top2 | 0.4964 | 0.4606 | 21/24 | 0.974 | 1305 | 1014 |
| hier-synth/top1 | 0.4820 | 0.4409 | 22/24 | 0.991 | 710 | 676 |
| hier-synth/top2 | 0.5647 | 0.5354 | 21/24 | 0.971 | 1290 | 981 |

Mechanism decomposition (recomputed from frozen spaces): routing recall is the
entire loss; the shortlist planner is STRONGER than flat's planner.

| arm | hit_rate | pos_acc\|hit | pos_acc\|miss |
|---|---|---|---|
| hier-desc/top1 | 0.433 | 0.909 | 0.000 |
| hier-desc/top2 | 0.567 | 0.812 | 0.000 |
| hier-synth/top1 | 0.543 | 0.812 | 0.000 |
| hier-synth/top2 | 0.713 | 0.751 | 0.000 |

MS-12 k-sweep (hier-synth/top2, everything else identical):

| k | acc | neg_acc | prompt_tok | ms | mean shortlist |
|---|---|---|---|---|---|
| 5 | **0.7014** | 22/24 | 2124 | 1440 | 67 |
| 8 | 0.5935 | 22/24 | 1506 | 1128 | 47 |
| 10 | 0.5647 | 21/24 | 1290 | 981 | 26 |
| 15 | 0.5360 | 23/24 | 1008 | 835 | 30 |
| 25 | 0.5827 | 20/24 | 644 | 659 | 18 |

k=5 decomposition: hit_rate 0.878 × pos_acc|hit 0.776 = 0.681 (= observed 0.681).
k=5/top2 beats flat on BOTH sub-metrics at 121 tools: pos 0.681 vs 0.665,
neg 22/24 vs 21/24. Paired McNemar flat vs k5 at 121: 26 vs 31 discordant,
p = 0.597 — statistically a tie.

MS-14 gradient (N=78 = github 62 + filesystem 14 + fetch 1 + sequential-thinking 1;
80% one server; 186 cases):

| arm | acc | pos_acc | neg_acc | prompt_tok | ms |
|---|---|---|---|---|---|
| flat@78 | 0.3602 | 0.3765 | 6/24 | 2870 | 1970 |
| hier k5/top2@78 | **0.5054** | 0.4568 | 20/24 | 1472 | 1040 |

flat@78 vs k5@78 McNemar: 19 vs 46 discordant, p = 0.0011 — the hierarchy's win
at 78 is significant. Composition is the driver, not size: with one near-duplicate-
heavy server at 80% of the manifest, flat selection collapses (negatives 6/24;
non-github positives 0.211) while the coarse gate holds the domain boundary
(non-github positives 0.474, neg 20/24).

Determinism probe (MS-14): two independent temp-0 flat runs on byte-identical
prompts: 0/186 verdicts differ — same-prompt temp 0 is deterministic on this
server; arm differences above are configuration properties, not run noise.

Flat error anatomy at 121: 85/254 misses, 42 from github's near-duplicate block;
misses skew toward indirect phrasings (planner nulls); 3 neg misses
(17×23 → get-sum; two → echo).

## Hypotheses (MS-7) — adjudicated

- H1 (hier-synth/top2 ≥ flat): FAIL at registered k=10 (0.5647 < 0.6835); PASS
  at the MS-12 k=5 point (0.7014 ≥ 0.6835), but the margin is within noise
  (p = 0.597) — a tie with 43% fewer prompt tokens and 35% less latency.
- H2 (flat tokens ≥ 5× hier): FAIL at k=5 (1.76×); holds only for the finest
  clustering (k=25: 5.8×) where accuracy is far below flat.
- H3 (synth pools > description pools): PASS in both topologies (top1 0.482 vs
  0.432; top2 0.565 vs 0.496) — synthetic-utterance bootstrap is worth keeping.
- H4 (gap grows with tool count): measured at two points only; the gap moves in
  the hierarchy's favor as composition degrades, and INVERTS significantly at
  the 80%-one-server point (p = 0.0011). Direction confirmed: hierarchy pays
  as the manifest gets harder, and dominant-server flat manifests are toxic.

Adoption rule (H1 + neg_acc within 2 points of flat): SATISFIED by
hier-synth/top2 at k=5.

## VERDICT

Build MCP tool selection hierarchically — but COARSELY. The measured optimum is
a gate over FEW domains (4–8 clusters; k=5 won, every k ≥ 8 lost to flat), a
top-2 union shortlist (mean 67 of 121 tools at k=5 — the gate is a boundary,
not a needle-finder), synthetic-utterance centroids (H3), and the LLM planner
over the shortlist with the {"tool": null} escape (91.7% null precision at
k=5/top2 vs 87.5% flat). Fine unsupervised hierarchies are the failure mode:
routing recall dies (43–71% at k=10) while the planner-given-hit stays at
75–91% — keep near-duplicate disambiguation in the LLM, never in the embedding.
The gate's biggest measured win is robustness to manifest composition (a
dominant server collapses flat selection to 36% and bleeds nulls; the gate
holds +14.5 points, p=0.001), which is exactly the risk profile of real MCP
setups where one big server (github-like) joins a handful of small ones.
Cost at 121 tools: one utterance embedding (~55 ms) + microsecond readouts;
shortlist prompts 57% of flat's tokens, 65% of its latency.

Implementation notes for wiresense MCP support (not registered, derived):
cluster per connected server-set with the k≈4–6 rule, re-embed on tools/list
change (MS-3 pools make this cheap), keep plan-twice (Rev G) on delegate picks,
and keep the production behavioral root untouched — MCP tools hang under the
existing tool path.

Scored data: results.jsonl (run 2: flat carried from run 1 + 4 hier arms),
results_run1_invalid_hier.jsonl (MS-11, archived invalid), sweep_k{5,8,15,25}_
valid.jsonl (MS-12/13), grad78_*.jsonl (MS-14). Invalid MS-12 first sweep kept
off-repo (byte-identical to k=10, superceded).

## Amendment MS-15 (registered pre-data): tool-selection-as-a-tool

Question: should "search for tools" be exposed to the planner as a tool — as the
PRIMARY selector (tiny manifest, model calls mcp_search when a task smells
tool-shaped), or as an ESCAPE HATCH inside the measured k=5/top-2 gate (planner
may call mcp_search when the shortlist doesn't fit)? Same frozen corpus, eval,
pools, budget C=200, planner params, k=5 synth space as run 2. New results file;
existing results untouched.

Search mechanics (both arms): query embedded via the production path
(norm_router_text, slot 6), cosine vs all 121 per-tool synth centroids, top-8
tools returned verbatim (name + desc[:C] + required args). Re-plan is
HISTORY-FREE (Rev G spirit): build_prompt over the SAME utterance with a
directive presenting the returned tools and the null escape; NO nested search;
exactly one search per case maximum. A non-search tool name returned from the
search-only manifest scores as a wrong answer directly (no recovery round).

| arm | first directive | search | scored |
|---|---|---|---|
| search-primary | only the mcp_search line + 3-way contract | on mcp_search | final pick |
| hier-synth-esc/top2 | k=5 top-2 shortlist + mcp_search line + 3-way contract | on mcp_search | final pick |

Metrics as MS-6, plus per-arm: search invocation rate, mean completions calls per
case, end-to-end ms and total prompt tokens (all calls summed), and for the
escape arm: pos_acc on gate-miss cases (run-2 baseline: 0.000 — the dead mass
this arm exists to recover).

Registered predictions (to be checked, not assumed):
- P1: search-primary lands BELOW both flat (0.6835) and gate k=5 (0.7014) on
  overall accuracy, with worse null discipline than gate (0.917) and roughly
  2x gate latency.
- P2: gate+escape recovers a meaningful share of the 12% miss mass
  (pos_acc on miss cases > 0.3, vs 0.000) and finishes >= gate k=5 overall,
  at <= 1.2 mean calls per case.
If P2 holds and null discipline holds >= gate - 2 points, the escape line
joins the implementation spec; if P1 fails, primary-as-tool is recorded as
rejected at this scale.

### Amendment MS-15a (registered pre-data, from canary): prime contract tightening

Pre-run canary (6 cases, diagnostic only, MS-10 precedent): the search-primary
contract as first written produced {"tool": null} on 6/6 cases INCLUDING both
positives — the model never invoked mcp_search. As written the arm would measure
that phrasing's null bias, not the search mechanism. Registered fix, exact
wording (replaces CONTRACT_PRIME in run_search.py):

  "Choose a tool for this request. You have no tools listed yet. If this
  request could be handled by a tool, call mcp_search first. Reply with ONLY
  a JSON object: {\"tool\": \"mcp_search\", \"args\": {\"query\": \"<a few
  keywords>\"}} to search for tools, or {\"tool\": null} if this request is
  pure conversation needing no tool. No other text."

If the tightened contract STILL nulls positives at scale, that is the recorded
answer for search-primary (a primary search tool a 4B planner won't invoke is
functionally dead). Also registered: missing/empty query arg in an mcp_search
call falls back to the case utterance verbatim; a second mcp_search after the
re-plan is not executed (one search per case) and scores as a wrong pick.
Escape-path mechanics verified offline pre-data (no eval cases): query
"convert 3pm Tokyo time to UTC" -> top-8 NN returns convert_time,
get_current_time at ranks 1-2; history-free re-plan picks convert_time with
correct args (340 prompt tokens).

### Amendment MS-15b (registered pre-data, from canary): dead search path in runner

Second canary pass caught a runner bug of the MS-13 class: the search-execution
branch was gated on `arm != "search-primary"`, so the search-primary arm
recorded got="mcp_search" WITHOUT ever executing the search (calls=1). Zero
scored data existed. Fix: the mcp_search branch now runs for both arms; canary
records wiped; re-canary must show calls=2 / searched=true on search-emitting
cases before the full run. No other change.

### Amendment MS-15c (registered pre-rerun): primary-arm scorer invalidated the arm

Post-run decomposition replay caught a scoring bug of the MS-11 class: the
MS-15 clause "a non-search tool name returned from the search-only manifest
scores as a wrong answer directly" was implemented as `ok = False` for EVERY
real-tool pick in the search-primary arm — including picks made AFTER a
legitimate search, which is the arm's core path. Evidence: 3 replayed cases
with want inside the returned top-8 re-picked the correct tool with args, and
their stored records show got == want yet ok = false. All 278 search-primary
records are INVALID (ok/schema_ok fields wrong; raw model outputs were valid
but args were not stored, so offline re-scoring cannot restore schema_ok).
Adjudication rule: archive the invalid records, fix the clause to apply ONLY
to blind picks (search not executed), rerun search-primary only. The
hier-synth-esc/top2@k5 arm never enters that branch (arm equality check) — its
278 records and all decomposition numbers stand as valid. Search-trigger stats
from the invalid run (91.4% search rate on the tightened contract) remain
descriptive evidence about the contract, not scores.

## MS-15 Results (runs of 2026-10-06; primary rerun under MS-15c)

Reference rows from run 2 / MS-12 sweep, same 278-case frozen eval:

| arm | acc | pos | neg(null) | schema | search rate | calls | pt | ms |
|---|---|---|---|---|---|---|---|---|
| flat (ref) | 0.6835 | 0.6654 | 21/24 | — | 0 | 1.00 | 3739 | 2205 |
| gate synth/top2 k5 (ref) | 0.7014 | 0.681 | 22/24 | — | 0 | 1.00 | 2124 | 1440 |
| search-primary | 0.4928 | 116/254=0.457 | 21/24 | 112/116 | 0.914 | 1.91 | 474 | 928 |
| hier-synth-esc/top2@k5 | 0.6583 | 164/254=0.646 | 19/24 | 159/164 | 0.219 | 1.22 | 2246 | 1563 |

Process note: three canary passes before data caught (a) a prime contract that
nulled 6/6 (MS-15a), (b) a dead search branch (MS-15b), and (c) a scorer that
invalidated the primary arm post-run (MS-15c, archived + rerun). Decomposition
numbers below come from the valid records.

Search-primary decomposition — the loss is retrieval, not picking:
- 246/254 positives searched (the tightened contract fixed triggering at scale;
  canary n=6 had suggested inversion — it was noise).
- Model-written queries surface the target in the returned top-8 only 51%
  (125/246). ok|target-in-top8 = 113/125 = 0.904; ok|target-missing = 3/121.
  The same model picking from the gate's ~67-tool shortlist scores 0.776; from
  an 8-candidate list, 0.904 — smaller menus pick better, consistently with
  the run-2 thesis. The arm dies because 2-4-word queries lose ~37 points of
  retrieval recall vs embedding the utterance itself (87.8%).
- Null discipline: 21/24; the 3 failures are all search-then-pick on chat.

Escape-arm decomposition — the option itself is not free:
- Escape fired on 61/278 turns (21.9%): 45 times on routing-hits (target
  already shortlisted; trigger miscalibrated) and only 15/31 on true routing
  misses (48% trigger recall on exactly the dead mass it exists for).
- Recovery on the 31 dead-mass positives: 5/31 = 0.161 overall (run-2 baseline
  0.000); per fired true miss 5/15 = 0.33. Real but small.
- When the search result DID contain the target, pick quality was unchanged
  (20/26 = 0.77 = gate's planner-given-hit) — the pick stage is not the loss.
- Contract tax: on the 217 paired turns where NEITHER arm searched, plain k5
  was right on 11 turns the escape variant lost and lost 3 it gained. Merely
  offering the escape (one extra manifest line + 3-way contract) costs ~4
  accuracy points on turns that never use it. Overall paired vs k5: 26 vs 14,
  McNemar p = 0.0807 (negative direction, not significant at 0.05).

### P1/P2 adjudication
- P1 (primary below flat and gate on accuracy): CONFIRMED — 0.4928 vs
  0.6835/0.7014. Null discipline worse than gate: CONFIRMED (0.875 vs 0.917).
  "Roughly 2x gate latency": REFUTED — it is the fastest arm (928 ms; tiny
  prompts), just inaccurate.
- P2 (escape recovers >0.3 of dead mass, finishes >= gate k5, <= 1.2 calls):
  FAILED on all three — 0.161 recovered, 0.6583 < 0.7014, 1.22 calls.

### VERDICT
Neither search arm joins the implementation spec. Search-as-primary is
rejected at this model scale: the planner writes lossy queries, and the
retrieval recall it needs (87.8%) is exactly what the utterance embedding
already delivers — the k5 gate IS the search, keyed on the best available
query (the utterance). The escape hatch is rejected as a default: per-fire
recovery on true misses (0.33) is real but the trigger fires on the wrong
turns (45 hits vs 15 misses) and the standing option tax (~4 pts on
non-escape turns) outweighs the recovered mass. If routing recall is ever the
binding constraint again, the lever is a better utterance-side retriever, not
model-mediated search.

## Amendment MS-16 (registered pre-data): implementation checks for production

MCP support is being implemented per the MS-12/15 verdicts. Two changes touch
measured ground, so both are registered before the corresponding data exists.

(a) NAMESPACING. Production manifests show MCP tools as `{server}__{tool}`
(collision-free across servers and against the native six; matches the corpus
and synth-pool keys). The frozen 278-case eval measured BARE names. Check:
rerun the adopted arm (hier-synth/top2, k=5, top-2) with every manifest line's
tool name prefixed `{server}__{name}`; scoring strips one trailing `__`-suffix
before comparing to want. Routing is untouched (utterance-keyed). Adopt the
prefixed form only if overall accuracy is within noise of the bare-name
0.7014 (paired McNemar, no significant regression at 0.05); otherwise fall
back to bare names + connect-time collision rejection. Runner flag:
MCP_PROBE_PREFIX=1; results to a NEW file; existing results untouched.

(b) PER-SERVER DEPLOYMENT GATE. A server is enabled in production config only
after its own frozen eval passes: cases generated from the server's real
tools/list with the registered MS-4 recipes (8 direct + 8 indirect positives
per tool capped at 64 total, hand negatives reused), frozen at
tools/mcp_eval/evals/<server>.json BEFORE any scoring, run on the SAME planner
and the SAME selector mode production will use (flat for <=20-tool servers,
gate otherwise). Bar: pos_acc >= 0.60 AND neg_acc >= 20/24. Failing servers
stay disabled; results append-only; frozen evals never edited. Regeneration
of a server's eval (e.g. its tools/list changed) requires a new protocol
amendment naming the reason.

### MS-16a result (full 278, run of 2026-10-06)

Prefixed names: acc 0.6763 (bare 0.7014), pos 168/254, neg 20/24, schema
165/168. Paired vs bare: bare-only-right 19, prefixed-only-right 12, McNemar
p = 0.2810 — no significant regression at 0.05. The model returned the
prefixed form in 278/278 scored picks (zero bare-name replies): the namespace
is contract-stable at this scale. VERDICT: adopt `{server}__{tool}` in
production manifests, per the registered rule. (An earlier read of the first
206 records, before the run finished, showed 0.6942/p=0.23 — noted only to
flag that partial-file adjudication happened once and was discarded unacted;
the verdict above is the complete-run number.)

### Amendment MS-16c (registered pre-scoring): per-server eval generator retry

The first gen_eval.py pass froze incomplete evals: 4/41 tools produced
unparseable generations (diagnosed verbatim: the model emits EOS mid-JSON —
finish=stop at 151 tokens with no closing brace — a seed-2 quirk on specific
tool cards; one character of description difference flips it). Zero cases
were scored. Remedy, registered before any data: gen_eval.gen retries ONCE at
seed+1 on parse failure (the same 2-attempt pattern as every other llama
call in this repo); the four incomplete freeze files are archived to
evals/_invalid_gen/ and all four servers regenerated + refrozen pre-scoring.
No other recipe change.

## Implementation (entries AT-BB, 2026-10-06)

The measured verdicts are now running code. Opt-in like face: a `"mcp"` key in
web/config.json (gitignored; web/config.example.json documents the format),
absent/disabled leaves both MCP globals None and the manifest byte-identical.

- `web/mcp_client.py` — stdio transport + lifecycle. One MCPServer per config
  entry (spawn with own process group, 2024-11-05 handshake, id-matched
  newline-JSON-RPC, tools/call -> joined text). Blocking reads run under
  to_thread; a wedged server is unblocked by killing its process group.
  Secrets: config env values must be "$ENV_NAME"; literals for
  TOKEN/KEY/SECRET/PASSWORD keys are refused at load. Dead servers: lazy
  reconnect on the next call plus a 60 s supervisor task; a deliberate
  kill() clears the respawn cooldown. Smoke: 9/9 corpus servers connect,
  8/9 tool counts exact, live reconnect mid-session verified.
- `web/mcp_gate.py` — the coarse gate, port of the frozen probe. RECIPE_A
  pools verbatim (temp 0.8 seed 1, build_prompt prefix), slot-6 embeddings
  via norm_router_text, k-means VERBATIM port (an argmin/argmax init
  divergence the equivalence check caught: 30/30 shortlists now match the
  recorded run-2 routes), top-2 union, C=200 lines, `{server}__{tool}`
  namespacing (MS-16a-adopted: 0.6763 vs 0.7014 bare, p=0.281, contract
  stable 278/278). Index cached in web/mcp_index.json keyed on a tools/list
  content hash; a 60 s watcher rebuilds when a server changes its tools.
  Flat manifest at <= 20 MCP tools (registered threshold); credential-bearing
  schemas refused at load; write-heuristic confirm flag per tool with
  per-server override.
- `web/server.py` — integration surface only: main() connects + builds the
  index (to_thread); tool_flow swaps `tool_manifest_directive()` for
  `MCP_GATE.directive_for(utterance)`; validate_tool delegates to the gate's
  inputSchema check (required args, types, caps, unknown keys dropped);
  run_tool calls tools/call (30 s default, result truncated to 300 chars,
  honest failure lines); risky MCP tools park for the spoken read-back via
  the existing pending machinery ("Just to confirm — run <name>?").
- `tools/mcp_eval/` — MS-16b deployment gate. gen_eval.py generates per-server
  frozen evals from the LIVE tools/list (RECIPE_B8 = MS-4 recipe B widened to
  the registered 8+8, seed 2, one retry at seed+1 after the diagnosed
  EOS-mid-JSON quirk, MS-16c); run_eval.py scores through the PRODUCTION gate
  over the combined server set; results append-only.

First deployment-gate run (bar: pos >= 0.60 AND neg >= 20/24):
time PASS (0.719), everything PASS (0.656), filesystem FAIL (0.422,
read/write near-duplicate confusion), git FAIL (0.531, log/show/add
disambiguation). Per registration, filesystem and git stay DISABLED. Note on
negatives: 3 of the 4 neg failures per server are the model correctly picking
native tools (timer/reminder/note are in the combined manifest); true null
discipline is 23/24 everywhere. The bar's 20/24 was set knowing this.

Live smoke (tools/mcp_smoke.py, fake WS, no audio path): direct call with
spoken confirmation, risky write parked + executed on YES, chat falls
through, mid-session server kill reconnects transparently. SMOKE PASS.

Not enabled anywhere yet: config.example defaults off. The user's live
config.json is untouched; the running web server still has pre-MCP code
until its next restart.

Hygiene note (post-BB, entry BC): the MS-16a prefixed rerun had also
overwritten tools/mcp_probe/summary.json in the working tree (summarize()
writes the default path unless MCP_PROBE_SUMMARY is set — same clobber class
as the search-arm rerun). No data or verdict was affected (adjudication reads
results_prefix.jsonl); the prefixed summary is preserved as summary_prefix.json
and summary.json restored to the committed run-2 state.

## Entry BD (registered before implementation): remote transport — Streamable HTTP, no auth

User ask: support remote MCP servers. Scope registered BEFORE code:

- Transport: Streamable HTTP (spec 2025-03-26) — POST one endpoint,
  JSON-RPC; server answers either `application/json` or an SSE stream on
  the same response; session id via the optional `Mcp-Session-Id` response
  header, echoed back on later requests; DELETE ends the session. Legacy
  2024-11-05 HTTP+SSE is NOT implemented (fallback only if a real server
  demands it).
- Auth DEFERRED (user decision): no headers/bearer/OAuth in this entry.
  Only no-auth remote servers can connect; anything requiring a token
  fails honestly at initialize. When auth lands it will reuse the
  $ENV_NAME expansion rule.
- Non-goals held: no GET SSE channel (server-initiated notifications),
  no streaming results (read the full body, then parse).

Design: `url` in a server's config selects remote mode; everything above
the transport (tools/list contract, gate, namespace, confirm policy,
supervisor/cooldown state machine) is unchanged — ZERO server.py edits
expected. Remote `kill()` = best-effort DELETE + clear session; a mid-call
404 (server restarted / session expired) gets ONE transparent re-init +
retry, then the existing honest-failure line. `alive()` = session open
and not down.

Verification (in order, all before any enablement):
1. Hand-rolled stateful streamable-HTTP server (stdlib, thread) in
   tools/mcp_http_smoke.py: initialize+session-id, tools/list, tools/call,
   one SSE-response call, session invalidation mid-session (client must
   re-init transparently), DELETE honored.
2. Regression: existing stdio smoke (tools/mcp_smoke.py) re-run must still
   pass unchanged.
3. Real-world probe: https://mcp.deepwiki.com/mcp (no auth, stateless,
   SSE-response): initialize, tools/list, one real call. Logged, not
   frozen — deployment-gate (MS-16b) applies before any real enablement.
Prediction: no gate impact of any kind (tools/list content is
transport-independent); new failure modes are network-level only.

### Entry BD results (run of 2026-10-06)

- tools/mcp_http_smoke.py (hand-rolled stateful streamable-HTTP server,
  stdlib thread, driven MCPServer-direct): SMOKE PASS 8/8 — initialize +
  session-id issue, JSON tools/call round-trip, MCP-Protocol-Version header
  after negotiate, SSE-response call parsed, isError raised as MCPError,
  mid-session 404 -> transparent re-init + retry (server saw exactly 2
  initializes), kill() -> DELETE seen, fresh restart.
- Regression: tools/mcp_smoke.py (stdio, live llama + gate) re-run
  unchanged: SMOKE PASS all four scenarios.
- Real-world probe (logged, not frozen, per registration):
  https://mcp.deepwiki.com/mcp — initialize 200 with SSE body, NO session
  id (stateless server; client handles: alive() keys on the handshake flag,
  not the id), protocol negotiated 2025-03-26; tools/list = 3 tools
  (ask_wiki_question, read_wiki_contents, read_wiki_structure);
  read_wiki_structure(fastapi/fastapi) returned 1818 chars in 0.8 s.
- Prediction held: zero server.py/mcp_gate.py edits (grep-verified — no
  references to argv/proc outside the client); gate untouched because
  tools/list content is transport-independent.

deepwiki is NOT enabled anywhere (MS-16b deployment gate still applies
before any real enablement; if it's ever enabled, its frozen eval gets
generated from the live tools/list like any stdio server). Auth remains
deferred: config has no headers field yet; a 401/403 at initialize fails
honestly with the "auth required? (deferred, entry BD)" hint.

## Entry BE (registered before any camera contact): probe onvif-mcp (user-supplied third-party server)

User ask: test the MCP implementation against github.com/qianlima365/onvif-mcp
and their ONVIF camera (creds known to the user; supplied per-call by the
probe harness, never through the model).

Source review BEFORE running (disclosed in full in the probe report):
- every tool takes host/usr/pwd AS ARGUMENTS — real voice use would push
  the camera password through the planner and (on risky tools) the spoken
  readback; conflicts with the house secrets rule. Probe passes creds in
  the harness call only.
- snap_image_to_minio uploads camera frames to MINIO_ENDPOINT and returns
  a URL via MINIO_PROXY_ENDPOINT (example defaults: third-party cloud
  ai.isstech.com) — NOT CALLED, ever.
- get_camera_list / get_camera_live hardcode a third party's creds
  (admin:qwer1234@10.17.20.110) and an external stream URL — NOT CALLED.
- handle_ptz_control physically moves the camera; focus_move likewise
  (imaging service). NOT CALLED without an explicit user ask. Note:
  handle_ptz_control does NOT match the RISKY_RE heuristic by name;
  focus_move does ("move") — recorded for the report.
- minio.py prints MINIO_SECRET_KEY to stdout at import (= the stdio
  JSON-RPC channel); only reachable via the snap tool, so moot here.
- handle_discover_devices is read-only WS-Discovery (also publishes a
  harmless dummy service on the LAN) — allowed.

Whitelist for this probe: tools/list; get_deviceInfo, get_rtsp,
handle_discover_devices (read-only, against 192.168.110.83 or LAN-wide for
discovery). Everything else refused by the probe script. Launcher:
`uv run --directory <repo> main_stdio.py` (stdio transport; --directory
keeps the client unmodified — no cwd support added for one probe).

No MS-16b deployment gate in this entry: enablement is a separate decision
given the findings; if wanted, the registered gate runs on the frozen eval
generated from the live tools/list like any other server.

### Entry BE results (run of 2026-10-06, camera 192.168.110.83)

- OUR CLIENT: clean pass. stdio connect 0.4 s, 8 tools listed, three
  whitelisted calls round-tripped, server process group killed at exit.
  First third-party stdio server beyond the probe corpus.
- THE SERVER'S stdio entry is broken: main_stdio.py re-creates
  `FastMCP('onvif-mcp')` AFTER `from main import *`, orphaning all real
  tools — it serves only a do-nothing `onvif_tool` stub. A 2-line launcher
  (/tmp/onvif-mcp/run_stdio.py: `from main import mcp;
  mcp.run(transport='stdio')`) serves the real 8; repo unmodified
  otherwise.
- CAMERA (ONVIF on :5000, confirmed by ws-discovery XAddrs; RTSP on
  :8554): get_rtsp returned 2 streams (H264 2304x1296, PROFILE_0/…) —
  fully working data. get_deviceInfo answered (valid ONVIF round trip)
  but with ALL FIELDS NULL — some cameras report empty DeviceInfo; the
  pipeline faithfully relayed the camera's answer, not a client bug.
  handle_discover_devices: 3.2 s, found the camera plus its own dummy
  published service.
- SCHEMA FINDING (the creds-as-args design, now measured): only `host` is
  required on get_deviceInfo/get_rtsp — usr/pwd carry defaults
  ('admin'/'admin'). A voice planner would pass host and credentials
  would either default wrong or have to be MODEL-FILLED; and get_rtsp
  embeds user:pass in the returned URI, which run_tool would speak aloud
  in the reply. Two leak paths through the model/TTS, as predicted.
- Probe script /tmp/probe_onvif.py (creds inline, deliberately NOT in the
  repo tree). Repo clone at /tmp/onvif-mcp (uv.lock synced).

VERDICT: implementation verified against a real camera; onvif-mcp AS-IS is
not shippable under house rules. If camera control is wanted for real: a
thin wrapper server (env-bound ONVIF_HOST/ONVIF_USER/ONVIF_PASS at spawn —
the $ENV_NAME rule; read-only tool subset; snapshot to local file, not
MinIO; PTZ behind the confirm park with per-server "confirm": true, since
"handle_ptz_control" does not match RISKY_RE by name). Wrapper would be a
separate registered entry.

## Entry BF (registered before actuation): user-authorized PTZ/focus attempt

User explicitly asked to test camera control. Sequence, gentlest first:
1. handle_ptz_control(direction="Left", speed=5) — if PTZ existed, the tool
   itself auto-stops after 3 s at 5% speed; expected per the null PTZ node
   in get_rtsp: server raises "device does not support PTZ control".
2. focus_move(speed=0.2) → wait ~2.5 s → focus_move(speed=0.0) to halt.
   DISCLOSED BUG: the tool's documented stop (speed=None) is unreachable
   through MCP — the schema type is strict float, so None fails validation;
   speed=0 (motor commanded to zero) is the only in-protocol stop. If the
   camera lacks an imaging service, OnvifClient construction fails first
   and nothing actuates.
Both calls go through the same client path as entry BE (whitelist extended
by this registration only). Every outcome recorded honestly, including
"camera moved" if it does.

### Entry BF results (run of 2026-10-06, camera 192.168.110.83)

- handle_ptz_control(Left, 5%): MCPError in 0.2 s — server raised
  "当前设备不支持云台控制" (device does not support PTZ control). No
  movement.
- focus_move(0.2 → 0.0 halt): MCPError in 0.3 s — ONVIF imaging service
  answered "The requested VideoSource does not support imaging settings".
  Motor never started; the halt leg was unnecessary.
- Error path validated end-to-end for real: FastMCP wraps tool exceptions
  as isError content; our client surfaced the device's own error text as
  MCPError within 0.3 s.

CONCLUSION: the camera is a fixed unit — no pan/tilt (no PTZ node) and no
motorized focus/zoom (imaging refused per-source). This server offers it
nothing beyond get_rtsp (working) / discovery (working) / deviceInfo
(nulls quirk); the actuation tools can never do anything on this hardware.
If streams-in-voice is the goal, the wrapper entry shrinks to: env-bound
creds + get_rtsp + local snapshot (HTTP digest fetch of GetSnapshotUri,
no MinIO), MS-16b gate before enablement.

## Entry BF2 (registered before actuation): raw ContinuousMove against the hollow PTZ stub

Capability diagnostic (read-only, direct ONVIF, bypassing the MCP server):
device DOES advertise PTZ XAddr (.../onvif/ptz_services) but GetNodes=[]
and GetConfigurations=[] and neither profile carries a PTZConfiguration —
user's Tuya hypothesis confirmed: motors are Tuya-protocol-driven, the
ONVIF layer is a stub. Also: GetDeviceInformation all-null confirmed
direct (camera quirk, not the MCP path); GetServices errors
(Missing element IncludeCapability — shim's schema sloppiness); imaging
service present but refused per-source earlier.

Now testing whether the stub is dead or merely unadvertised: raw
ptz.ContinuousMove(ProfileToken=PROFILE_0, PanTilt x=0.1 ~10% speed,
y=0, Zoom 0) for ~1.5 s, then Stop. Outcomes recorded honestly either
way; if it moves, a wrapper could drive PTZ via raw ONVIF despite the
empty node registry.

### Entry BF2 results (run of 2026-10-06)

- PTZ GetServiceCapabilities: responds (endpoint alive) — EFlip/Reverse
  False, nothing move-related enabled.
- RAW ContinuousMove(ProfileToken=PROFILE_0, PanTilt x=0.1, 1.5 s):
  SOAP Fault **"PTZ Profile Not Exist"** before any motion. The stub is
  dead, not merely unadvertised — the firmware itself refuses PTZ on a
  real video profile.

FINAL: three independent confirmations the camera's ONVIF cannot steer it
(0 nodes/0 configs; no profile binding; firmware faults on raw move).
Tuya app drives the motors via Tuya's own protocol. Voice PTZ for this
device would require the Tuya side (tinytuya local control with the
device's local key), or a firmware/ONVIF toggle if the camera's settings
expose one — not an ONVIF wrapper. Streams + snapshot wrapper remains the
only shippable ONVIF shape for this camera.

## MS-17a (registered before any generation): mid-loop gate keying — continuation text vs utterance text

Design context (session discussion, no code yet): the deep loop's per-step
decisions (need a tool? which one?) are candidates for the tree/gate
machinery as a System-1 layer. Load-bearing assumption: the gate's
shortlist, keyed on the MODEL'S CONTINUATION TEXT (what it says/thinks
right before acting) rather than the user utterance, still recalls the
right tool. This entry measures exactly that, retrieval-only — no loop
changes, no need-check classifier (that is MS-17b).

Generation (RECIPE_C, verbatim):
  'Write 8 different short things an assistant might SAY or THINK
  mid-answer, right before it should use this tool. One or two sentences
  of the assistant's own reasoning or speech that leads naturally into
  the call; never the tool name; max 24 words each. Reply with ONLY
  JSON: {"utterances": ["...", ...]}'
- 24-word cap vs RECIPE_A's 12 is deliberate: continuations are
  reasoning-shaped sentences, and length/register drift is part of what
  is being tested.
- Mechanics mirror the frozen pool gen exactly (card Name/Server/
  Description/Required args; build_prompt([], card, RECIPE_C); max_tokens
  220, temp 0.8, id_slot 5); seed 3 (unused: pools seed 1, eval seed 2);
  parse via mcp_gate.parse_json_obj, >=5 utterances else ONE retry at
  seed 4 (MS-16c rule), else empty list + recorded failure. All 121
  corpus tools. Output frozen to tools/mcp_loop/continuations.json;
  the generator refuses to overwrite.

Index: rebuilt STRICTLY from the committed synth_pools.json via the
probe's own code path (run_eval: embed+norm_router_text slot 6,
mean-of-L2, committed kmeans, MCP_PROBE_K=5 asserted) — the shipped gate
recipe, unchanged. NO continuation text enters the index (no-leak rule;
that would be an index-recipe change = MS-18, separately registered).

Scoring (retrieval-only; recall = want in top-2 cluster-union shortlist):
- Set A (baseline): the 242 frozen eval positives, utterance-keyed,
  recomputed on THIS build.
- Set B (treatment): every frozen continuation text, keyed to its tool.
- Secondary: mean shortlist size both sets; per-tool recall deltas;
  utterance-vs-continuation shortlist Jaccard per tool.

Bars (set BEFORE data): primary recall_B >= 0.80; non-inferiority
recall_B - recall_A >= -0.05. Both pass -> MS-17b (need-check readout)
is worth registering. recall_B collapses (< recall_A - 0.10) ->
continuation keying is dead; the loop keeps marker legs only. Middle
band -> candidate remedy is RECIPE_C-augmented pools, but that is an
index-recipe change: MS-18, separately registered, never a silent edit.

Files: tools/mcp_loop/{gen_cont.py, run_17a.py, continuations.json,
results_17a.jsonl, summary_17a.json}. Runners committed before data.

### MS-17a results (run of 2026-10-06)

Generation: 968 continuation texts over all 121 tools, seed 3, ZERO
retries and zero failures (the MS-16c rule never fired). Texts read as
intended — assistant-voice reasoning leading into a call ("I need to save
this summary to the notes file").

Hygiene disclosure: the scorer crashed AFTER all 1210 rows were written
(summary-only code, `jac` initializer lost in an edit); the runner now
resumes from complete rows (deterministic frozen inputs) and the
aggregation was re-run from them. Rows themselves were untouched.

Primary numbers (index rebuilt strictly from committed pools, k=5
asserted; cluster sizes [5, 46, 29, 3, 38]):
- recall_A (utterance-keyed, 242 frozen positives): 0.8884 — reproduces
  the committed ~0.88 recall; build sanity holds.
- recall_B (continuation-keyed, 968 texts): 0.8058.
- delta_pooled −0.083; delta_per_tool −0.083; mean shortlist 70.9 (A) vs
  74.4 (B) tools; utterance-vs-continuation shortlist Jaccard 0.601.

ADJUDICATION against the registered bars:
- Primary (recall_B >= 0.80): PASS, by 0.006 — statistically AT the bar
  (n=968, binomial SE ~0.013), not comfortably above it.
- Non-inferiority (delta >= −0.05): FAIL (−0.083).
- Collapse band (recall_B < recall_A − 0.10 = 0.788): NOT entered.
Registered middle band, so: the candidate remedy (RECIPE_C-augmented
pools — an index-recipe change) is NOT applied; it goes to MS-18 as a
separate registration if pursued. No silent edits.

Per-tool shape: 56 tools degrade, 19 improve, 45 flat (n_A=2/tool —
per-tool deltas are ±0.5-noisy; trust the aggregate). Worst: jargon-heavy
GitHub-class tools (simulate-research-query, get_job_logs, PR-comment
family) and convert_time/list_notifications.

Design reading for the System-1 loop: continuation keying WORKS at a
usable level — 0.81 shortlist recall is far above the model's own
retrieval (0.51 target-in-top8, MS-15) with zero generation cost, and the
0.60 Jaccard confirms shortlists genuinely differ mid-loop (re-keying on
state is necessary, not optional). The open question MS-18 would answer:
does an index built from both text kinds close the −0.08?

## MS-17b (registered before any generation): the need-check readout — {tool, continue, done} over model-state text

Design context: the System-1 loop's first classifier decides, at a round
boundary, what the model's just-produced text means: a tool is needed
(tool), reasoning is mid-flight (continue), or the answer is complete
(done). This entry measures whether the DEPLOYED tree mechanism can make
that call — before any loop wiring exists.

Classes and data:
- tool: MS-17a's frozen RECIPE_C texts (968, tools/mcp_loop/
  continuations.json) — reused verbatim, NOT regenerated.
- continue: RECIPE_D (verbatim): 'Write 8 different short things an
  assistant might say or think MID-ANSWER while reasoning from what it
  already knows, where it does NOT need any tool, lookup, or outside
  information. It is working on this tool's topic but should NOT use the
  tool or mention it. One or two sentences; never the tool name; max 24
  words each. Reply with ONLY JSON: {"utterances": ["...", ...]}'
- done: RECIPE_E (verbatim): 'Write 8 different short COMPLETE final
  answers (1-2 sentences) an assistant might give a user on this tool's
  topic, fully resolving the question from known information WITHOUT
  calling this tool or any other; never mention the tool. Max 30 words
  each. Reply with ONLY JSON: {"utterances": ["...", ...]}'
- RECIPE_D/E: per-tool conditioning over all 121 corpus tools (domain-
  matched hard negatives), card/generation mechanics identical to
  MS-17a (build_prompt, 220 max_tokens, temp 0.8, id_slot 5); continue
  seed 5, done seed 7, ONE retry at seed+1 each (MS-16c rule), >=5
  utterances per call, frozen to tools/mcp_loop/state_texts.json
  ({"continue": {...}, "done": {...}}), generator refuses overwrite.

Split rule (registered, no-leak): per key, list order as generated —
pool = texts[:n//2], eval = texts[n//2:]. Same split for the reused
RECIPE_C texts (0-3 pool / 4-7 eval).

Readout: the deployed tree mechanics VERBATIM (TreeWalker/DecideReadout):
embed norm_router_text(text) slot 6 -> arm-C head _project -> norm_rows;
per-class centroid = mean of pool vectors -> norm_rows; classify =
cosine argmax. No projection/training changes; no confidence gating
(tree-probe rule).

Eval: held-out halves only (484/class), balanced. Report confusion
matrix, per-class recall, overall accuracy. Known limitation, disclosed:
texts are synthetic approximations of the 4B model's in-loop register;
MS-19's canaries (2 live rounds per class, diagnostic-not-scored, the
MS-10 precedent) would catch register mismatch before any scored run.

Bars (set BEFORE data):
- overall accuracy >= 0.75;
- tool-class recall >= 0.80;
- false-tool rate (continue or done classified tool) <= 0.15.
All three pass -> MS-19 loop wiring is justified. tool recall passes but
false-tool fails -> candidate remedy is a two-stage readout (binary
tool/no-tool, then done/continue) = MS-17c, separately registered. tool
recall fails -> the need-check is dead; the loop keeps model-driven
markers only.

Files: tools/mcp_loop/{gen_state.py, state_texts.json, run_17b.py,
results_17b.jsonl, summary_17b.json}. Runner committed before data.

## MS-17b (results + adjudication)

Run mechanics, two crashes disclosed (neither touched scored data):
1. `NameError: time` during centroid building — the runner had a
   `__import__("time").time()` initializer but a bare `time.time()` in
   the progress print. Crash BEFORE any row was written; fixed with a
   normal import; the retry was a clean full run (no rows existed to
   resume or discard).
2. `TypeError` in summary aggregation — `sum()` over a generator of
   `dict_values` objects. Crash AFTER all 1448 rows were written and
   flushed; fixed; the retry took the registered resume path
   (aggregate-only, row count validated == 1448, no re-embedding).

Results (results_17b.jsonl, 1448 held-out rows; summary_17b.json):

```
              pred tool  cont  done | recall
true tool          252   129   103 |  0.5207
true continue      123   244   113 |  0.5083
true done           71    83   330 |  0.6818
overall accuracy 0.5704    false-tool rate 0.2012
bars: overall>=0.75 FAIL, tool-recall>=0.80 FAIL, false-tool<=0.15 FAIL
```

Adjudication (registered fork, applied verbatim): tool-class recall
fails -> the need-check readout is DEAD. No two-stage rescue is
registered (that fork required tool recall to pass). The deep_turn loop
keeps its model-driven markers (IR_TOOL / BROWSE_TOOL) as the sole
loop-continuation signal; no centroid readout at round boundaries.

Reading (diagnostic, not scored): the confusion is symmetric
tool<->continue (~25% each way) and above chance (0.57 vs 0.33) — the
arm-C head space encodes topic/intent, not "does this text need a
tool"; synthetic state text of all three classes projects into
overlapping regions. This is a structural miss, not a marginal one:
any rescue would be a TRAINED readout head (the arm-C precedent —
supervised, labeled rounds), a different mechanism class with its own
registration, not another centroid/pool recipe. Not justified by
current evidence.

Consequence for the System-1 loop design: what survives measurement is
entry-point tool selection (MS-17a: utterance-keyed readout 0.89/0.81 —
already the shipped gate mechanism). Mid-loop system-1 gating is
rejected by MS-17b. MS-18 (RECIPE_C-augmented pools) remains open as a
selector-quality fork only; it does not rescue this verdict.

Housekeeping: `snapshot.json` (tinytuya device dump from the BF2
diagnostics) sits untracked at the repo root; left alone deliberately —
it may hold device local keys and must not be committed.

## MS-18 / MS-19 closure (decision recorded, no data generated)

- **MS-19 (System-1 loop wiring: need-check readout at round boundaries
  + mid-loop tool selection re-keyed on continuation text): CLOSED —
  premise dead.** MS-17b failed its registered bars (tool recall 0.5207
  vs 0.80); the registered fork applied verbatim: the deep_turn loop
  keeps its model-driven markers (IR_TOOL / BROWSE_TOOL) as the sole
  continuation signal. There is no readout left to wire. Resurrection
  path, for the record: a TRAINED round-classification head (the arm-C
  precedent — supervised over labeled live rounds), which is a
  different mechanism class and would need its own registration from
  scratch. Not justified by current evidence.
- **MS-18 (RECIPE_C-augmented pools index): MOOT — its consumer is
  gone.** Its sole purpose was closing the utterance->continuation
  recall gap (0.888 -> 0.806, Jaccard 0.60) for continuation-keyed
  queries against the gate index. The only component that ever issued
  continuation queries was MS-19's mid-loop selector; with MS-19
  closed, no runtime path queries the gate with model-state text.
  Entry selection runs on utterances (recall 0.8884, shipped mechanism)
  and needs no augmentation. Resurrection path: any future design that
  feeds model-state text to the gate index.

The frozen MS-17a/17b data (continuations.json, state_texts.json,
results, summaries) stays committed and is reusable by either
resurrection path without regeneration.

## MS-16b failure autopsy (post-hoc analysis of frozen data — no new eval data)

Question: filesystem (pos 0.422) and git (pos 0.531) fail the deployment
gate; time (0.719) and everything (0.656) pass. WHY — gate routing or
planner selection?

Method (read-only replay): spawned the same four servers, loaded the
index from the hash-matched cache (b6ac5cf5 — built during the scored
run; "INDEX SOURCE: hash" confirmed identical centroids/assignment), so
`shortlist(utterance)` reproduces the exact manifest the planner saw.
Recomputed shortlists for all 224 pos cases, joined with the frozen
results rows. Replay artifacts in /tmp (replay_gate.py, replay_16b.jsonl,
fail_dump.txt) — diagnostic only, matching the probe precedent; the
committed record is this entry.

Routing recall (want ∈ top-2-union shortlist): filesystem 62/64, time
32/32, everything 59/64, git 56/64 — 210/224 overall (93.8%).

Failure decomposition (mechanical, from the join):
- filesystem: 37 fails = 35 selection + 2 routing; 16 nulls (14 routed)
- git: 30 fails = 22 selection + 8 routing; 16 nulls (8 routed)
- time: 9 fails = 9 selection + 0 routing; 5 nulls (5 routed)
- everything: 22 fails = 17 selection + 5 routing; 12 nulls (8 routed)

Finding 1 — the gate is exonerated. When a case fails, the wanted tool
was ON the planner's manifest 79/94 times; routing misses are 15. The
MS-12/MS-15 selector delivers at deployment scale. What fails is the
4B planner's pick among visible siblings (or its null escape).

Finding 2 — most selection "failures" are eval artifacts (qualitative
read of all 94, my post-hoc judgment, NOT a scored measurement):
- Deprecated wants: filesystem read_file says "DEPRECATED: Use
  read_text_file instead" — 3 frozen wants; an ideal planner preferring
  the canonical tool scores 0 on all three.
- Defensible siblings: "Show me the commit history here." (want
  git_show, model's git_log at least as correct); "List everything in
  the Music folder." (want list_directory_with_sizes, plain
  list_directory answers the user); "Can you remind me when it's 5 PM?"
  (want time get_current_time, the model's native `reminder` pick is
  the right assistant behavior); "Execute the research query now."
  (want everything__simulate-research-query, the model picked the REAL
  native ir_research).
- Utterances without the discriminator: the show/diff/log triangle
  ("What changes are in this commit?"), staged vs unstaged idioms.
- Arg-grounding nulls: the eval is history-free, but utterances like
  "read my notes file" carry no path; the production null contract
  fires, and in production that is a clarifying turn, not a failure.
- Demo-tool wants (everything server): get-annotated-message,
  toggle-subscriber-updates, get-resource-reference are MCP demo
  fixtures with demo descriptions.

Adjudicated tally: ~75/94 failures eval-borne or defensible; ~19
genuine planner failures, concentrated in the filesystem edit/write
family (8 of fs's 16 nulls are edit/write utterances — the model
under-triggers on mutations), git jargon ("stage" 5/5 -> git_status,
"check out" -> git_branch), and verbatim-request nulls (convert_time,
directory_tree). Under a hypothetical perfect planner (only the ~19
fixed), filesystem would score 34/64 (0.53 — STILL failing), git
44/64 (0.69 — passing), time 24/32, everything 44/64. The filesystem
failure is substantially eval-borne: its frozen set wants a deprecated
alias and ambiguous listing variants.

Schema discipline where the planner did pick: 100% (scored run's
schema_ok).

## MS-16b-2 (registered before any generation): eval-recipe amendment B9

Scope: change the EVAL GENERATION only. No gate code, no planner, no
model, no description rewriting (desc rewriting is treatment — its own
registration if ever). Bars and adjudication rule unchanged:
pos >= 0.60 AND neg >= 20/24; PASS enables the server per the original
MS-16b rule.

Amendments (both registered here, before any new data):
1. RECIPE_B9 (verbatim) amends B8 with self-containment + uniqueness:
   'Write 8 short things a user might say to a voice assistant for this
   tool. "directs": plain direct requests for what the tool does, do
   not copy the tool name, max 14 words each. "indirects": goal-oriented
   phrasings where the tool is implied but never named, max 16 words
   each. Two rules: every utterance must be self-contained — include
   the concrete details the tool needs (file or folder names, content
   to write) so nothing needs a follow-up question; and phrase it so
   THIS tool is the only reasonable choice among similar tools —
   include what makes it distinct (sizes, recursion, staging,
   conversion, whatever separates it). Reply with ONLY JSON:
   {"directs": ["...", ...], "indirects": ["...", ...]}'
2. Tool-level exclusion (mechanical, desc-based): a tool whose
   description contains "DEPRECATED" is excluded from eval generation
   (an alias of a canonical sibling; a case can only invite the wrong
   pick). Applies to filesystem read_file. Demo/simulation tools are
   NOT excluded — the everything server's content is what it is.

Mechanics unchanged: one call per tool, build_prompt card format, temp
0.8, max_tokens 560, seed 2 with one retry at seed+1 (MS-16c), id_slot
5; frozen to evals/<server>.b9.json BEFORE any scoring, refuses
overwrite, pos stride-sample cap 64, same 24 hand negatives verbatim.
Runner: eval/results/verdict paths take the suffix from MCP_EVAL_SUFFIX
(default empty = B8 files, byte-identical behavior). Scoring through
the production gate over the COMBINED four-server manifest (the B8 run
was combined — cross-server picks in the frozen rows confirm it; B9
must match). All four servers regenerate: a recipe change re-freezes
the eval set, and the two passing servers must face the same recipe
for like-for-like comparison. The B8 frozen evals and results stay
committed and untouched.

Files: gen_eval.py + run_eval.py amendments committed before data;
evals/<server>.b9.json; results/<server>.b9.{jsonl,verdict}.

## MS-16b-2 (results + adjudication)

One launch hiccup, disclosed: the first background launch passed
MCP_EVAL_SUFFIX=b9 (no dot), giving evals/filesystemb9.json ->
FileNotFoundError. No cases were scored; relaunched with MCP_EVAL_SUFFIX=.b9.
Runner code unchanged (registered contract: suffix splices into
{name}{SUFFIX}.json; default empty = B8 paths).

Results (320 cases = 224 pos + 96 neg; combined four-server manifest;
"index from cache (hash b6ac5cf5)" — the SAME index the B8 run scored
against; no pool regeneration):

```
server      B8 pos   B9 pos   neg     schema    verdict
filesystem  0.4219   0.6406   20/24   41/41     FAIL -> PASS
time        0.7188   0.7500   20/24   24/24     PASS
everything  0.6562   0.4375   20/24   28/28     PASS -> FAIL (flipped)
git         0.5312   0.6406   20/24   37/41     FAIL -> PASS
```

The autopsy's prediction held: with self-contained, uniquely-resolved
utterances and the deprecated alias excluded — same gate, same planner,
same index, zero code change — filesystem (+0.22) and git (+0.11) clear
the registered bar. The B8 failures were substantially eval-borne.

The everything flip is a true measurement, not a recipe artifact. Its
tools are MCP SDK demo fixtures (echo / annotated-message / resource
links / long-op trigger / simulated research); under the uniqueness
rule the frozen cases demand demo-fixture precision, and the model
repeatedly does the CORRECT assistant thing: it picks the real native
research tool over simulate-research-query (4x) and refuses demo-fixture
utterances (trigger-long-running-operation 6x null). Demo tools do not
map to distinct user intents; the B8 pass was vagueness dressed as
coverage. everything stays gated off — correct for a server that exists
to exercise protocol features, not serve users.

Residual concentrations (recorded for the future, all within PASSing
servers): git git_commit -> native note (3x) and the show/log/diff
triangle persisting despite uniqueness phrasing (the three descriptions
genuinely overlap); git 4 schema misses (right tool, required arg
absent — 37/41); filesystem edit_file -> read_text_file (2x);
time get_current_time -> null / ir_research / convert_time ("what time
is it in X" vs conversion ambiguity persists).

Deployment state per the registered rule (PASS enables): filesystem,
git, and time are CLEARED FOR ENABLEMENT; everything remains disabled.
The user's live config.json is untouched — flipping the enable switch
is the user's call. When enabled with <= 20 MCP tools total the gate
runs flat; the four-server set (41 tools) runs through the cached index.

Files: tools/mcp_eval/results/{filesystem,time,everything,git}.b9.{jsonl,verdict}.

## INCIDENT (2026-10-07): the gate ground flaps — unpinned npx/uvx spawns

During the memory-feature deployment gate (PROTOCOL-memory.md B4a) the
filesystem re-run FAILED (pos 0.5625, bar 0.60) against the registered
0.6406 PASS, while git (0.7031) and time (0.7500) re-ran IDENTICAL to
registration — four verdict rows each, byte-stable. The flapping is in
the ground, not any code change:

- The gate cache is keyed on a content hash of the spawned tools/list
  (registered ground: 473e375b). During investigation, back-to-back
  spawn sets produced the cached hash on one run and a fresh rebuild on
  the next; an instrumented replication produced yet another payload
  hash. Fresh rebuilds are deterministic per tool set (rerun3 ≡ rerun4
  byte-identical, 88+64+88 rows), so a rebuild itself proves the spawned
  tool set differed from the cached one.
- npm and PyPI show no new releases (filesystem 2026.8.31, time/git
  2026.8.18), so the flap is between locally cached package versions and
  the registry's current resolution: `npx -y` / `uvx` re-resolve against
  registry metadata whose local TTLs expire, then swing back.
- On the new (freshly-resolved) ground filesystem not only loses 5 pos
  cases, it gains schema failures — the newly-resolved server version
  changed tool schemas. The registered 0.6406 was measured on the older
  cached ground and does not transfer.

Consequence for the registered policy (PASS enables, verdicts are
version-scoped): filesystem is currently enabled on a ground where its
deployment gate FAILS. Per policy it should be re-cleared or parked;
disabling it is a config edit — the user's call.

Resolution path (recommended, not yet applied — web/config.json is the
user's live file): pin each MCP server version in the spawn command
(e.g. `npx -y @modelcontextprotocol/server-filesystem@<ver>` /
`uvx mcp-server-time==<ver>`), then re-run the three frozen gates on the
pinned ground as the new registered verdicts. Pinning makes the cached
index and every future spawn agree by construction, which is what the
cache hash assumed all along.

## CORRECTION (2026-10-07, later the same day): the incident above is WRONG about the cause

The "version flap" did not happen. npm and uv caches hold exactly one
version of each server across both days (filesystem 2026.8.31, time/git
2026.8.18; no newer releases exist), and the hash differences that
motivated the incident entry were artifacts: ad-hoc payload hashes
computed with a different scheme than the gate's `_signature()`, plus the
cache file being overwritten between checks.

The real ground difference — found by comparing per-case prompt_tokens
(registered runs ~1470 tokens uniform; re-runs 659-989 variable) and
confirmed by the registered run log (/tmp/ms16b2_run.log) — is the TOOL
UNIVERSE:

- Registered MS-16b-2 (Oct 6): `run_eval.py filesystem time git everything`
  — all FOUR servers connected: 41 MCP tools, gate over 41.
- B4a re-runs (Oct 7): `run_eval.py filesystem git time` — 3 servers,
  28 tools, gate over 28. That IS the production set (everything was
  never enabled in config), but it is NOT the ground the registered
  verdicts were measured on.

Decisive replication (results/*.b941.*, the frozen .b9 evals byte-copied
under a diagnostic suffix, all four servers connected, current code):
filesystem pos 40/64 (0.6250) PASS — versus 36/64 (0.5625) FAIL on 28
tools. git 45/64 (0.7031) and time 24/32 (0.7500) are IDENTICAL on both
grounds — git's +4 over registration is entirely the repo_path pin
defaulting (0862a91), not the universe change. everything 0.4219 FAIL,
consistent with registration.

So the deployment-gate verdicts are UNIVERSE-scoped: the k=5/top-2
clustering reshuffles when the connected set changes, and filesystem —
14 of 28 tools, half the universe — loses ~4-5 pos cases at 28 tools that
it passes at 41. The registered clearance certified a ground that
production never runs.

Consequences (supersede the incident entry's resolution path):
1. Version pinning remains good hygiene but fixes nothing here; the pin
   recommendation is withdrawn as a remedy for THIS failure.
2. On the production ground (28 tools): git PASS (0.7031), time PASS
   (0.7500), filesystem FAIL (0.5625 < 0.60). Per the PASS-enables policy
   filesystem is out of compliance on the ground it actually runs in.
   Options are the user's: park filesystem, or register and measure the
   unmeasured 21-40-tool band (flat-over-28 as a candidate arm — the
   frozen recipe measured flat only up to 20 tools), then re-freeze.
3. Policy amendment for all future gates: a deployment gate must run over
   the exact enabled production server set, never a superset.

## REGISTERED (2026-10-07): flat-28 arm for the unmeasured 21-40-tool band

Motivation (same-day correction above): the frozen recipe measured flat
only up to 20 tools (MCP_FLAT_MAX=20) and the gate beyond — never the
21-40 band. Production runs 28; filesystem fails the gate there (0.5625)
while passing richer-manifest grounds. The candidate is flat.

Arm: MCP_FLAT_MAX treated as 28 for the measurement (runtime override in
a diagnostic wrapper; the frozen constant is untouched pending
adjudication); the exact production server set (filesystem, git, time =
28 MCP tools); frozen .b9 evals byte-copied to diagnostic suffix .f28;
current code (repo_path defaulting, memory feature) — identical to the
gate-28 re-runs, so this isolates the selector.

Pre-committed decision rule (registered before any data):
- ADOPT MCP_FLAT_MAX=28 iff filesystem pos >= 0.60 AND git pos >= 0.60
  AND time pos >= 0.60, each with neg >= 20/24, on the frozen evals.
- On adoption: constant committed, production server restarted, capability
  suite re-run as the regression check (27/27 expected).
- If any bar fails: NO adoption; git/time stay on the gate at 28 (they
  already pass it); parking filesystem is recommended to the user (config
  edit on their call). The flat-28 numbers are recorded either way — the
  band is then measured, negative result included.
- No bar changes, no eval edits, no new selector machinery in this arm.

Baselines for comparison (already measured, same code, gate-28):
fs 0.5625 FAIL / git 0.7031 PASS / time 0.7500 PASS, neg 21/24 each.

## RESULTS (2026-10-07): flat-28 ADOPTED

Measured on the frozen .b9 evals (byte-copied to .f28), the exact
production set, current code — versus the gate-28 baselines (same code,
same ground, this morning's re-runs):

```
           gate-28          flat-28           bar    verdict
filesystem 0.5625 FAIL      0.6094 (39/64)    0.60   PASS
git        0.7031 PASS      0.6875 (44/64)    0.60   PASS
time       0.7500 PASS      0.7500 (24/32)    0.60   PASS
neg        21/24 each       21/24 each        20/24  all PASS
```

Decision rule (pre-committed before data, above): all three clear ->
ADOPT. MCP_FLAT_MAX is now 28, committed with this section. Production
restarted on the flat selector; the capability suite re-run is the
regression check.

Honest margins, recorded as measured: filesystem clears by exactly ONE
case (38/64 = 0.59375 would fail) — its compliance at this ground is
fragile, and run-to-run noise has been +-1 case all week. git gives one
case back to the flat manifest; time is untouched. The 29-40 band remains
unmeasured — the gate still owns >28, and the registered fallback if
filesystem drops below the bar on any future ground change is parking it.

Runner note (disclosed): the first wrapper launch silently adjudicated
the OLD unsuffixed results — MCP_EVAL_SUFFIX unset meant
results/<server>.jsonl + evals/*.json, the MS-16 originals (fs 0.4219,
the number that motivated this whole investigation) — instead of running
the arm. The .f28 run is the genuine arm ("88 cases to run (0 done)" in
the log). Second time the SUFFIX/resume interaction has manufactured a
plausible-looking number from stale rows; the earlier instances are in
the correction above.

Regression check (registered adoption step): capability suite run 7 on
the flat-28 selector — **27/27 PASS**, including the mid-session server
reconnect scenario (time self-healed during the run). Production server
restarted on the flat manifest (28 tools + native).

Files: tools/mcp_eval/results/{filesystem,git,time}.f28.{jsonl,verdict};
evals/*.f28.json are byte-copies of the frozen .b9 evals (the diagnostic
suffix the runner requires); tools/mcp_eval/run_flat28.py is the
registered wrapper (runtime MCP_FLAT_MAX override, frozen constant
untouched during measurement).

## MS-20 (registered pre-data): beam-2 path scoring for the gate shortlist

Question (2026-10-07): the k=5/top-2 shortlist is an UNORDERED union (mean 67
of 121 tools, 2124 prompt tokens); the WireScale beam-2 result and the
production router-walk adoption (`tools/eval_router_beam.py`, root 57/62 vs
greedy 56/62) both say: keep two coarse candidates alive and combine levels
with a length-normalized mean path cosine. Can ranking the (cluster, tool)
paths by mean(cluster_sim, tool_sim) — and truncating — hold accuracy at a
fraction of the shortlist budget?

Arms (all prefixed names, MS-16a form; frozen hier-synth space k=5; frozen
pools `synth_pools.json`; C=200 from `C_budget.json`; planner params
byte-identical to run 2 (temp 0, id_slot 5, max_tokens 120, history-free);
frozen 278-case eval; NEW results file, existing results untouched):

| arm | clusters | membership | line order | lines |
|---|---|---|---|---|
| union (baseline) | top-2 | union | registry order | all (~67) |
| beam-union | top-2 | union | mean-path desc | all |
| greedy16 | top-1 | in-cluster | tool-sim desc | 16 |
| beam16 | top-2 | union | mean-path desc | 16 |
| beam32 | top-2 | union | mean-path desc | 32 |

Path score for tool t in cluster c: mean(cos(c, utt), cos(t, utt)). The
router-walk lesson applies verbatim: a (cluster, tool) path has two terms and
a bare tool score one — raw sums structurally outbid shorter paths (measured
43/62 on the router), so the mean, never the sum.

Route records log gold rank-in-shortlist per case (None when absent), so
hit_rate and pos_acc|hit decompose per arm offline, same decomposition form
as run 2.

Bars (registered before any scored run):

- Sanity: the union arm must reproduce the MS-16a prefixed baseline 0.6763
  within ±0.005 (same-prompt temp-0 determinism was probed in MS-14) AND
  neg_acc ≥ 20/24. Otherwise STOP — no verdict on drifted ground.
- Primary adoption (web/mcp_gate.py, default-on above MCP_FLAT_MAX, new
  registered constant MCP_TOPM): beam16 acc ≥ union acc AND beam16
  mean_prompt_tokens < 0.75 × union's AND beam16 pos_acc ≥ greedy16 pos_acc
  (beam beats greedy at equal budget — the mechanism check).
- Fallback (registered now, no post-hoc M shopping): if beam16 fails only
  the acc clause, beam32 may adopt under the same three clauses.
- Ordering-only (beam-union) is never an adoption candidate: same
  membership, same tokens, position effects fragile — report only.

Runner: `tools/mcp_probe/run_beam.py` (imports run_eval's loaders, kmeans,
embed, and planner verbatim; never edits them; own `results_beam.jsonl` +
`summary_beam.json`). MS-10 rules: this section + runner committed before
any scored run; 6-case canary green; one full pass in a single session;
failures recorded as failures.

## MS-20 (results + adjudication, run of 2026-10-07)

Single session, 278 cases per arm (6-case canary records included, run-2
practice). Machinery notes: the runner crashed at the sanity gate after the
union arm completed (string/int compare on the neg count) — fixed and
resumed; union's 272 fresh records carried over verbatim (resume-safe), the
6 canary records were always part of every arm's 278.

| arm | acc | pos_acc | neg | hit_rate | pos_acc\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity ✓ 0.6727 vs 0.6763 ±0.005) | 0.6727 | 0.6535 | 21/24 | 0.878 | 0.744 | 2265 | 1284 |
| beam-union | 0.6727 | 0.6535 | 21/24 | 0.878 | 0.744 | 2265 | 1282 |
| greedy16 | 0.5324 | 0.5000 | 21/24 | 0.606 | 0.825 | 567 | 542 |
| beam16 | 0.5971 | 0.5748 | 20/24 | 0.721 | 0.798 | 608 | 574 |
| beam32 | 0.6259 | 0.5984 | 22/24 | 0.776 | 0.772 | 1088 | 777 |

VERDICT: **KEEP UNION** — primary bar fails on the acc clause (beam16
0.5971 < union 0.6727; fallback beam32 0.6259 also short). web/mcp_gate.py
unchanged; the registered constant MCP_TOPM is NOT introduced.

Mechanism read (both pre-registered decomposition clauses landed):

- **Beam-2 replicated for the third time**: at equal 16-line budget, beam16
  beats greedy16 +6.5 acc pts, +11.5 hit_rate pts (0.721 vs 0.606) — keeping
  the runner-up cluster alive recovers most of the recall a greedy commit
  loses, same mechanism as WireScale E and the router walk (beam16 pos
  0.5748 ≥ greedy16 0.5000: the pos clause passed).
- **But the k=5 gate is already the right granularity**: truncation's
  recall loss (hit 0.878 → 0.721 at M=16) outweighs the planner's
  option-tax gain (pos|hit RISES as the manifest shrinks: 0.744 → 0.798 →
  0.825, consistent with MS-15). The coarse boundary wants its members.
- **Ordering is a no-op**: beam-union is byte-identical to union on every
  metric — the temp-0 planner did not change a single pick when the same 67
  lines were reordered by mean path score. As registered, never an adoption
  candidate; now also empirically dead.
- The Pareto point is real but unregistered: beam16 buys 3.7× fewer tokens
  (608 vs 2265) and 2.2× lower latency (574 vs 1284 ms) for −7.5 acc pts.
  Recorded for any future constrained-budget entry; NOT adopted here.

Scored data: `tools/mcp_probe/results_beam.jsonl`, `summary_beam.json`
(runner `run_beam.py`; sanity-gate crash fix committed with the results).

## MS-21 (registered pre-data): binary-split walk shortlist ("split to <10")

Question (2026-10-09, user proposal): split the tool population in two,
score each half, keep the higher-scoring half, split again, stop when the
group holds fewer than 10 — does that walk beat the shipped gate?

Construction: balanced binary tree over the SAME synth-pool centroid space
as the gate (8 utts/tool, l2(mean(l2)), the run_beam build verbatim).
Recursive split: project member centroids on their top principal axis,
partition at the median (deterministic, perfectly balanced, no empty
nodes; SVD sign ambiguity cannot change the partition). Stop at nodes
< 10 tools (121 → 61/60 → 31/30/30/30 → 16/15 … → leaves of 7–8; 4
commitment levels). Node scorers (both registered, user left the scorer
open): `max` = max cosine(query, member centroid); `cent` = cosine(query,
node centroid, node = l2(mean of members)).

Arms (frozen 278-case eval, prefixed names, same planner as MS-20):
- sanity: `union` rerun first — must reproduce acc 0.6763 ± 0.005 AND
  neg ≥ 20/24, else STOP, no tree verdict on drift.
- `bwalk-max` — the user's scheme: greedy, max scorer, leaf manifest.
- `bwalk-cent` — greedy, centroid scorer.
- `bbeam2-max` / `bbeam2-cent` — beam width 2, path score = mean of
  per-level child scores (equal depth ⇒ mean == sum; mean recorded for
  principle); candidates = union of surviving leaves.
- `bbeam4-max` — beam width 4 (~32 candidates).

Bars:
- MS-21a (replacement): best tree arm acc ≥ same-run union acc AND
  hit_rate ≥ 0.878 → gate replacement candidate. PREDICTION ON RECORD:
  FAIL — a 4-commitment walk needs ~97% per split to match union recall
  (0.878^(1/4) ≈ 0.968); measured single-commitment anchors are lower
  (5-way ≈ 0.61–0.72, 8-way 0.84–0.92).
- MS-21b (budget): bbeam2-max acc > beam16's registered 0.5971 at
  mean_prompt_tokens ≤ 669 (1.1 × beam16's 608) → recorded as the better
  constrained-budget shortlist (Pareto note only; the gate is unchanged
  unless MS-21a passes).
- bwalk arms descriptive, no bar.

Runner: `tools/mcp_probe/run_btree.py` (imports run_beam's
directive_for for byte-identical manifest format and the frozen run_eval
machinery; writes results_btree.jsonl / summary_btree.json; canary
MCP_BTREE_CANARY=1). This section + runner committed before any scored
run. On fail: KEEP UNION, record as the depth-compounding datapoint.

### MS-21 results (2026-10-09, scored; runner ce71a84)

Sanity: union reproduced 0.6727 / neg 21 (registered 0.6763 ± 0.005) —
drift-free; tree arms adjudicated.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (gate, sanity) | 0.6727 | 0.6535 | 21/24 | 0.878 | 0.744 | 2265 | 1268 |
| bwalk-max (user scheme) | 0.4784 | 0.4567 | 17/24 | 0.520 | 0.879 | 332 | 436 |
| bwalk-cent | 0.3957 | 0.3622 | 18/24 | 0.406 | 0.893 | 330 | 383 |
| bbeam2-max | 0.5432 | 0.5276 | 17/24 | 0.626 | 0.843 | 574 | 569 |
| bbeam2-cent | 0.4892 | 0.4685 | 17/24 | 0.563 | 0.832 | 573 | 546 |
| bbeam4-max | 0.6007 | 0.5827 | 19/24 | 0.736 | 0.791 | 1070 | 773 |

VERDICT: **KEEP UNION** (both bars fail).
- MS-21a: best tree arm bbeam4-max 0.6007 < union 0.6727 AND hit 0.736
  < 0.878 → not a gate replacement. The registered prediction (needs
  ~97%/split; anchors lower) is CONFIRMED and now measured: walk recall
  0.520 over 4 splits ⇒ per-split survival ≈ 0.520^(1/4) ≈ 0.85.
- MS-21b: bbeam2-max 0.5432 < beam16 0.5971 → no budget win either.

Mechanism reads:
- The scheme optimizes the step that was already strong and pays in the
  step that decides the outcome: pos|hit at the ~8-tool leaf is the best
  in the whole MS series (0.879 — the planner is near-perfect given the
  right 8 tools), but only 52% of golds survive the walk; 0.520 × 0.879
  = 0.457 pos_acc vs union's 0.878 × 0.744 = 0.653.
- max beats centroid as the node scorer (+12.4 recall pts greedy:
  0.520 vs 0.406) — "is anything relevant in this half" reads intent;
  the node centroid's mean smooths it away. Registered for any future
  tree work.
- Depth loses to breadth at equal budget: bbeam4-max (1070 tok) vs
  MS-20 beam32 (1088 tok): 0.6007 vs 0.6259; bbeam2-max (574 tok) vs
  beam16 (608 tok): 0.5432 vs 0.5971. Four sequential commitments with
  beam recover less than one 5-way cut with top-2 breadth. The deep
  narrow tree is dominated by the shallow broad gate everywhere it was
  measured.
- Negatives degrade with narrowness (walks 17–18/24 vs union 21/24): a
  plausible-but-wrong tool always survives in a tiny list, and the
  planner picks it instead of answering null.
- Cheapest measured shortlist remains ~332 tokens (bwalk-max) at 0.478 —
  recorded for completeness; nothing here displaces union at any point
  of the trade-off curve.

Scored data: `tools/mcp_probe/results_btree.jsonl`, `summary_btree.json`.
Gate unchanged (web/mcp_gate.py untouched).

## MS-22 (registered pre-data): sliding-window shortlist

Question (2026-10-09, user proposal): slide a 10-tool window across the
population (step 7 ⇒ overlap 3), score each window, and build the final
list from the top-scoring windows ("join the similar highest score").

Interpretation registered: the things scored and joined are WINDOWS;
window score = max cosine(query, member centroid) — the MS-21-winning
scorer. "Similar to the highest" is operationalized two ways: fixed
top-N windows (bars live here — budget-controlled) and a threshold rule
(descriptive: all windows with score ≥ top − 0.03, δ fixed pre-data).

Window order is a registered variable, two variants:
- `reg` — registry order (load_tools: sorted server files, tools in
  corpus order). Windows ≈ server-proximity bands, semantically arbitrary.
- `pc1` — tools sorted by projection on PC1 of the gate's synth-pool
  centroid space. Windows = 1-D semantic bands; isolates whether window
  value comes from the scan itself or from order that follows geometry.

Coverage rule (both orders): starts = range(0, n−9, 7); if the last
start ≠ n−10, append n−10 (all 121 tools covered, every window exactly
10 tools). Candidates = union of picked windows' members, shown in
registry order (order is a measured no-op).

Arms (frozen 278-case eval, prefixed names, same planner; sanity `union`
rerun first — acc 0.6763 ± 0.005, neg ≥ 20/24, else STOP):
- `win2-reg`, `win2-pc1` — top-2 windows (~17-20 candidates; budget arm)
- `win4-reg`, `win4-pc1` — top-4 windows (~40 candidates; coverage arm)
- `winthr-reg`, `winthr-pc1` — threshold rule (descriptive, no bar)

Bars:
- MS-22a (replacement): best barred arm acc ≥ same-run union acc AND
  hit_rate ≥ 0.878. PREDICTION ON RECORD: FAIL — registry windows
  ignore geometry; pc1 windows are 1-D bands, expected ≤ the k-means
  union's recall at comparable budgets.
- MS-22b (budget): win2-pc1 acc > beam16's 0.5971 AND tokens ≤ 669 →
  Pareto note only; gate unchanged unless MS-22a passes.
- reg-vs-pc1 ordering effect: descriptive.

Runner: `tools/mcp_probe/run_window.py` (same import discipline as
run_btree; writes results_window.jsonl / summary_window.json; canary
MCP_WIN_CANARY=1). Committed before any scored run.

### MS-22 results (2026-10-07, scored)

Run: 7 arms × 278 cases, run committed pre-data (b56a67f); canary clean
(6 cases, shortlist sizes as registered: win2 ≈ 17–20, win4 ≈ 31–37,
winthr 10–79, union 41/8); canary ledger deleted before the scored run.
Sanity `union` first: acc 0.6727 (registered 0.6763 ± 0.005 — inside),
neg 21/24 ≥ 20 → window arms ran.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | **0.6727** | 0.6535 | 21/24 | **0.8780** | 0.7444 | 2265 | 1269 |
| win2-reg | 0.5144 | 0.4843 | 20/24 | 0.6299 | 0.7688 | 682 | 553 |
| win4-reg | 0.6223 | 0.6102 | 18/24 | 0.8031 | 0.7598 | 1239 | 825 |
| winthr-reg | 0.6187 | 0.6024 | 19/24 | 0.8268 | 0.7286 | 1336 | 890 |
| win2-pc1 | 0.4460 | 0.4173 | 18/24 | 0.5157 | 0.8092 | 673 | 572 |
| win4-pc1 | 0.5180 | 0.4921 | 19/24 | 0.6496 | 0.7576 | 1204 | 814 |
| winthr-pc1 | 0.5683 | 0.5551 | 17/24 | 0.7283 | 0.7568 | 1348 | 909 |

Verdicts:
- **MS-22a: KEEP UNION.** Best barred arm win4-reg: acc 0.6223 < union
  0.6727 AND hit 0.8031 < 0.878 — fails both clauses. The registered
  prediction (FAIL) held, as in MS-20 and MS-21.
- **MS-22b: no budget win.** win2-pc1 acc 0.446 < 0.5971 (and 673 tokens
  misses the 669 bar as well — fails both).
- **reg > pc1 at every mode** (win2 0.514 vs 0.446; win4 0.622 vs 0.518;
  thr 0.619 vs 0.568). Reading: the registration called registry windows
  "semantically arbitrary", but the corpus is sorted by server, and
  servers are coherent domains — registry bands are server bands, i.e.
  a hidden content-defined partition. PC1 bands are genuinely geometric
  and lose: a 1-D contiguous band through the centroid space cannot
  reproduce the disjoint top-2 k-means union's coverage — it interleaves
  tools whose 2-D structure projects onto the same coordinate and cuts
  across the density structure. Third consecutive measurement of the
  same law: **the shortlist's recall comes from content-defined disjoint
  groups (k-means), never from position-defined contiguous ones**
  (MS-21 binary tree, MS-22 sliding windows), and the max-member scorer
  (MS-21's winner, used here verbatim) does not rescue a bad partition.
- Pareto notes, unadopted: winthr-reg (hit 0.827 @ 1336 tok) and win4-reg
  (0.803 @ 1239) sit near beam32 (0.776 @ 1088); none reaches union
  recall at any budget. Window machinery (17 windows/order, overlap 3)
  archived in results_window.jsonl.

MS-22 post-hoc descriptive cut (2026-10-07, same scored run): accuracy
given the gold tool IS in the shown shortlist ("if the window actually
contains the right answer"). Union 166/223 = 0.744; win2-reg 123/160 =
0.769; win4-reg 155/204 = 0.760; winthr-reg 153/210 = 0.729; win2-pc1
106/131 = **0.809** (best of all arms); win4-pc1 125/165 = 0.758;
winthr-pc1 140/185 = 0.757. So the sliding window loses entirely on
coverage (answer present 52–83% vs union's 88%), never on selection —
and a dose effect across window arms: ok-given-present falls
monotonically with shown size (≤20 tools 361/439 = 0.822; 21–40 344/453
= 0.759; 41+ 263/386 = 0.681). acc = hit_rate × ok|present: per-hit
varies only ~0.73–0.81 while presence varies 0.52–0.88, so the product
is dominated by recall. Smaller shortlists pick better when right but
 miss more; the ~41-tool union sits on the same per-hit curve.

MS-22 follow-up diagnostic (2026-10-07, user proposal: step 5 = overlap 5,
"every gold then sits in 2 adjacent windows"). Geometry confirmed: gold-tool
membership multiplicity jumps 0.406 → 0.949 (reg). But presence moves the
OTHER way (embeddings only, no planner; diagnostic validated by reproducing
the scored step-7 hit rates): top-2 reg 0.630 (step 7) → 0.571 (step 5);
top-4 reg 0.799 → 0.760; pc1 strictly lower throughout. Reading: selection
is a competition among windows, and denser tiling multiplies the rivals
(24 windows vs 17) more than the gold's tickets — the gold's best window
scores the same under any tiling (max member cosine), while every relevant
region now fields ~2 high-scoring windows instead of ~1, and one of the
two top-N picks gets spent re-confirming gold's twin window (mean
shortlist 18.0 → 14.8 candidates). Presence would need ≥ ~0.86 at per-hit
~0.78 to threaten the union's 0.673; denser tiling moves away from that.

MS-22 follow-up diagnostic 2 (2026-10-07, user proposal: wrap-around last
window = last 5 + first 5, "make the possibility even"). Embeddings only,
no planner. Presence, reg order: top-2 0.571 (append) → 0.583 (wrap);
top-4 0.760 → 0.772 — a +1.2 pt nudge, still ~30 pts under the union's
0.878; pc1 unchanged (0.492/0.622). The premise itself fails: conditioning
on gold-tool membership multiplicity under the wrap tiling, multiplicity-1
golds are NOT disadvantaged (top-2 presence 3/5 = 0.600 vs 145/249 = 0.582
for multiplicity-≥2; top-4 5/5 = 1.000 vs 0.767). A window containing gold
scores from the gold itself (max member), no matter how many other windows
also contain it — evenness of coverage buys nothing. Every tiling variant
measured (step 7/5, append/wrap, reg/pc1, threshold joining) lands in
0.49–0.83 presence vs the k-means union's 0.878: the partition, not the
tiling, is the binding constraint.

MS-22 follow-up diagnostic 3 (2026-10-07, user's race model: "the 2 slides
containing the answer should outscore the rest"). Step-5 wrap tiling, reg
order, embeddings only. Geometry confirmed (user's literal example: registry
position 32 = git_commit, window [25..34] = search_files..git_reset; 95% of
golds sit in exactly 2 adjacent windows). But the race outcome: gold's best
window ranks top-2 in 148/254 = 0.583; rank 3-4 in 0.193; rank 5-8 in
0.142; rank 9+ in 0.083. Failures are hair-thin (winner-minus-gold window
score: median 0.021, mean 0.028) yet real: "Get me the annotated message"
→ gold get-annotated-message (pos 1!) scored 0.771, rank 16, beaten by
get_notification_details 0.845 — the embedding genuinely prefers a wrong
tool. The δ=0.03 threshold arm rescues the hair-thin losses (hence 0.827
presence) but ~17% of cases lose by more than the band. Mechanism: window
score = best member's cosine, so each window fields a champion and 24
champions sit within hundredths of each other; "contains the answer" ≠
"highest cosine". The union avoids this race entirely — its coarse groups
put rivals (annotated-message, notification-details) in the SAME shortlist
and let the description-reading planner make the fine call; the window
makes that fine call early, with the weaker judge (cosine), before the
planner sees anything.

## MS-23 (registered pre-data): window-union combination ("safety net")

Question (2026-10-09, follow-up to MS-22): does unioning the k-means
gate's shortlist with sliding-window picks create a new Pareto point?
Pre-check on this thread (embeddings only, no planner): union ∪ top-2
windows lifts presence 0.878 → 0.917 at +4.4 tools; union ∪ threshold
windows (score ≥ top − 0.03) → 0.957 at +10.5 tools. Clusters and
windows miss different cases — windows rescue 20 of the union's 31
presence misses — but per-hit degrades with list size, so the acc
projection is a wide band (0.65–0.71 vs union 0.673 / flat 0.684).

Arms (frozen 278-case eval, prefixed names, same planner; sanity
`union` rerun first — acc 0.6763 ± 0.005, neg ≥ 20/24, else STOP):
- `uwin2` — union ∪ top-2 windows (registered MS-22 tiling: window 10,
  step 7, append rule; registry order; score = max member cosine)
- `uwinthr` — union ∪ {windows with score ≥ top window − 0.03}
Candidates = set-union of index sets, shown in registry order.

Bars:
- MS-23a (adopt as gate): best combo arm acc ≥ same-run union acc +
  0.02 AND neg ≥ 20/24. PREDICTION ON RECORD: borderline — presence
  math says possibly the best selector yet measured; the dose-effect
  prior says the bigger list gives the gain back (per-hit at ~80 tools
  is unmeasured). Union stays unless both clauses pass.
- MS-23b (Pareto note): best combo acc ≥ flat's 0.6835 AND tokens ≤
  0.85 × 3739 = 3178 → recorded as matching flat's accuracy at ~75% of
  its cost; gate still unchanged unless MS-23a passes.

Runner: `tools/mcp_probe/run_uwin.py` (same import discipline; writes
results_uwin.jsonl / summary_uwin.json; canary MCP_UWIN_CANARY=1).
Committed before any scored run.

### MS-23 results (2026-10-07, scored)

Run: 3 arms × 278; runner committed pre-data; canary clean (union sl
41/8; uwin2 added 5–17 window tools; uwinthr 2–38); canary ledger
deleted before the scored run. Sanity `union` first: 0.6727 (inside
±0.005), neg 21/24 → combo arms ran.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 2265 | 1285 |
| uwin2 | 0.6906 | 0.6772 | 20/24 | 0.9173 | 0.7382 | 2421 | 1393 |
| uwinthr | 0.6942 | 0.6850 | 19/24 | 0.9567 | 0.7160 | 2636 | 1481 |

Verdicts:
- **MS-23a: KEEP UNION, both arms.** uwin2 acc 0.6906 < union+0.02
  (0.6927) by 0.002; uwinthr passes the accuracy clause (+0.0215) but
  neg 19/24 < 20 — the registered guardrail fires by ONE case. The
  neg slide is real and monotone in list size (21 → 20 → 19): the
  safety net that rescues positives also adds tempting distractors.
- **MS-23b: FIRES for both arms — new Pareto points.** uwinthr acc
  0.6942 ≥ flat's 0.6835 at 2636 tokens (70% of flat's 3739); uwin2
  0.6906 at 2421 (65%). First selectors measured to beat flat's
  accuracy at less than flat's cost. Recorded; gate unchanged per bars.

Readings:
- The embeddings-only presence pre-check transferred exactly:
  predicted 0.917/0.957, measured 0.9173/0.9567. What only the scored
  run could price was per-hit at ~80 tools: 0.716, not the feared
  ~0.68 — the union core keeps selection sharp.
- The user's sliding-window idea, rejected as a standalone selector
  (MS-22), adds +2.2 pts as a safety net over the gate: clusters and
  windows fail on disjoint cases, and the combined pick set recovers
  20 of the gate's 31 presence misses for +371 tokens.
- Not adopted per registered bars (one-case neg margin on the best
  arm). A serving-variance rerun (MS-8 style, both records kept) can
  re-open the adoption question only with a fresh registration.

## MS-24 (registered pre-data): direct tool-score band selector (no clusters)

Question (2026-10-09, user proposal): drop the k-means folders entirely —
score each of the 121 tools against the embedded query and keep the tools
above threshold. Two operationalizations:
- absolute threshold τ: keep tools with cosine ≥ τ — pre-checked FAILED
  (gold cosines span p10 0.752 / median 0.846 / p90 0.921: no fixed τ
  works; τ=0.75 drops 10% of golds outright; τ=0.80 presence 0.740);
- relative band δ: keep tools within δ of the top scorer — pre-check
  (embeddings only): δ=0.05 presence 0.728 @ 13.5 tools; δ=0.08 presence
  0.866 @ 34.4 tools (p90 74, max 102 — fat tail); δ=0.12 0.953 @ 69.3.
  δ=0.08 dominates every measured selector at its size (beam-32 0.776
  @ 32; winthr-windows 0.827 @ 40). Reading: "everything within the
  embedding's resolution of the best candidate" = defer-to-planner rule
  with no clustering.

Arms (frozen 278-case eval, prefixed names, same planner; sanity `union`
rerun first — acc 0.6763 ± 0.005, neg ≥ 20/24, else STOP):
- `band08` — tools with tool-cosine ≥ (top-1 tool cosine − 0.08)
- `band08cap40` — band08, then if > 40, keep top-40 by cosine
  (tail cap; band membership never rank-cut below the band)
Candidates shown in registry order. Query embedded raw
(norm_router_text) — no rephrase stage; the synth pools already
rephrase on the tool side.

Bars:
- MS-24a (adopt as gate): best band arm acc ≥ same-run union acc +
  0.02 AND neg ≥ 20/24. PREDICTION ON RECORD: borderline — presence
  0.866 is 1.2 pts UNDER union's 0.878; adoption requires per-hit to
  beat the dose curve (~0.76 @ 34 tools) on list coherence, reaching
  ~0.80, and neg ≥ 20. acc projection band: 0.66–0.70 vs union 0.673.
- MS-24b (budget Pareto): best band arm acc ≥ union − 0.01 AND tokens
  ≤ 1585 (0.70 × union's 2265) → recorded as matching the gate at
  ~half its token cost; gate unchanged unless MS-24a passes.

Runner: `tools/mcp_probe/run_band.py` (same import discipline; writes
results_band.jsonl / summary_band.json; canary MCP_BAND_CANARY=1).
Committed before any scored run.

### MS-24 results (2026-10-07, scored)

Run: 3 arms × 278; runner committed pre-data; canary clean (band sizes
4–61, cap bit exactly at 40 on wide bands); ledger deleted before the
scored run. Sanity `union` first: 0.6727 (inside ±0.005), neg 21/24.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 2265 | 1286 |
| band08 | 0.6619 | 0.6575 | 17/24 | 0.8661 | 0.7591 | 1210 | 875 |
| band08cap40 | 0.6439 | 0.6417 | 16/24 | 0.8307 | 0.7678 | 941 | 752 |

Verdicts:
- **MS-24a: KEEP UNION, both arms.** band08 acc 0.6619 < 0.6927 and
  neg 17/24 < 20; band08cap40 0.6439 / 16/24 — strictly worse.
- **MS-24b: missed by ONE case.** band08 tokens 1210 ≤ 1585 passed
  with room; acc 0.6619 vs required 0.6627 — one more correct case
  and the budget bar fires. As measured, no budget win is recorded.

Readings:
- Presence transferred exactly for the third consecutive time:
  pre-check 0.866, measured 0.8661.
- Per-hit landed dead on the dose curve (0.759 at ~34 mean tools,
  curve says 0.759 for 21–40) — the list-coherence hypothesis did NOT
  beat size as the per-hit driver.
- The band's real cost is the no-tool arm: neg 17/24 (70.8%) vs
  union's 21/24 (87.5%). Mechanism: on no-tool queries nothing scores
  well, so "within δ of the top scorer" is a wide, shapeless net of
  mediocre tools (canary: 48–61 tools on negs) — the planner faces a
  menu of plausible-looking wrong answers with no coherent structure
  to support returning null. The union's coherent clusters anchor
  rejection; the band dissolves exactly when rejection matters.
- Status: band08 (0.662 @ 1210 tok) is recorded as an extreme-budget
  data point between beam16 (0.597 @ 608) and union (0.673 @ 2265);
  not adopted, gate unchanged.

MS-24 post-hoc cut (2026-10-07, same scored run): does band width signal
"no tool fits"? No — the correlation runs the WRONG way for a fix.
Band sizes: should-call median 30 / no-tool median 30 (no separation);
9/24 no-tool cases have band ≤ 20. And 6 of the 7 band08 no-tool
failures are NARROW bands (neg_06 band=1 → planner picked the only
tool shown; neg_10/16/13 all picked echo off bands of 7–15; neg_09
band=19), while wide-band negs mostly return null correctly (band ≥ 39:
5/6 correct). Reading: "stay close to the winner" anchors the list on
the top scorer, and on no-tool queries the top scorer is noise — a
tight band around noise looks decisive. The union cannot be fooled
this way because membership is folder-based, not winner-anchored.
Small-band strength on should-call cases and null weakness on no-tool
cases are the same property.

## MS-25 (registered pre-data): two-step gate — intent check, then band

Question (2026-10-09, user proposal): split the decision. Step 1 asks
only "tool or no tool?"; if a tool is needed, step 2 runs the winner
band (band08) and the planner picks. Directly targets the band's single
measured weakness (neg 17/24) while keeping its proven strengths
(presence 0.866, per-hit 0.759, ~34-tool lists).

Step-1 variant rejected pre-data (embeddings only): null-gating on the
top-1 tool cosine. ROC: best operating point τ=0.72–0.75 nets only
+3 (catches 8/24 negs, wrongly rejects 5/254 should-call cases) — the
no-tool top-1 median (0.792) sits just under the should-call median
(0.877); confident noise is indistinguishable from intent by cosine.
Step 1 must read language, not similarity.

Step 1 (registered): blind intent call — build_prompt([], utterance,
INTENT_DIRECTIVE), temp 0, max_tokens 4, NO manifest, no tools shown.
Instruction asks for exactly one word: TOOL (the assistant should call
a tool to fulfill this) or CHAT (plain conversation, no tool needed).
CHAT → final {"tool": null}, planner never called. TOOL → band08
shortlist → planner as registered in MS-24.

Arms (frozen 278-case eval, prefixed names; sanity `union` rerun first
— acc 0.6763 ± 0.005, neg ≥ 20/24, else STOP):
- `step1-band08` — the two-step pipeline above. prompt_tokens recorded
  = intent tokens + planner tokens (full pipeline cost); intent
  decision and confusion recorded per case.

Bars:
- MS-25a (adopt as gate): acc ≥ same-run union acc + 0.02 AND
  neg ≥ 20/24. PREDICTION ON RECORD: borderline-positive — deciders
  are R (should-call cases the blind intent check wrongly sends to
  CHAT; expected small, 2–8, worst on indirect/trap phrasings) and N
  (negs caught; unknown, LLM-dependent). acc = band08's 167 pos-level
  − R + N → projection 0.67–0.69 at ~1,300 mean tokens. Union stays
  unless both clauses pass.
- MS-25b (budget Pareto): acc ≥ union − 0.01 AND tokens ≤ 1585 →
  recorded as gate-level accuracy at ~55% of gate tokens. Note: band08
  scored 184/278; the bar needs ≥ 185, i.e. N − R ≥ 1.

Runner: `tools/mcp_probe/run_step1.py` (imports frozen machinery +
band_indices from the committed run_band; writes results_step1.jsonl /
summary_step1.json; canary MCP_STEP1_CANARY=1). Committed before any
scored run.

### MS-25 results (2026-10-07, scored)

Run: 2 arms × 278; runner committed pre-data; canary clean (already
showed the echo cases going to CHAT); ledger deleted before the scored
run. Sanity `union` first: 0.6727 (inside ±0.005), neg 21/24.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 2265 | 1283 |
| step1-band08 | 0.5468 | 0.5079 | **23/24** | 0.6220 | **0.8165** | **786** | 800 |

Verdicts:
- **MS-25a: KEEP UNION.** acc 0.5468 << 0.6927. MS-25b also fails
  (0.5468 < 0.6627) despite 786 tokens (28% of gate cost).

Readings:
- The registered R prediction was WRONG and is disclosed: predicted
  2–8 blind rejections, measured **62/254** (24%) should-call cases
  sent to CHAT (chat_exits 82, of which 20 negs). "Does this need a
  tool?" read blind, the model says CHAT on a quarter of real tool
  requests — tool intent is much less lexically obvious than assumed.
- The null half worked beyond hope: **neg 23/24**, the best no-tool
  score ever measured in this program (union 21/24), at zero manifest
  cost. CHAT exits also cut latency (800 vs 1283 ms mean).
- Conditional on intent saying TOOL, the pipeline is excellent:
  per-hit 0.8165 — above every unconditional arm (dose curve ~0.76 at
  this size); stating "a tool is needed" is itself information that
  sharpens picking.
- Design lesson: null-detection and tool-selection have different
  error profiles (false CHAT vs false TOOL), and a hard two-step gate
  pays the full cost of step-1 false negatives. A soft cascade
  (step 1 only biases the planner, never exits early) is the natural
  follow-up but is NOT registered here.
- Status: gate unchanged; 786-token/23-null data point archived.

### MS-26 registration: `no_tool` as a listed option (menu framing) — 2026-10-09, pre-data

Question (user proposal): the gate only builds the menu; abstention is the
planner's call and the JSON contract already allows `{"tool": null}` — so the
band's neg failures are planner decisions, not menu defects. Does *how the
abstain option is offered* change those decisions? Hypothesis: a listed
`no_tool` entry ("none of the above") competes for the pick with the same
salience as real tools, while the parenthetical null clause is an escape hatch
the planner under-weights; `null` also carries no description to match against,
whereas tool picks are largely description-matching.

Manipulation (registered verbatim): planner directive unchanged EXCEPT one
extra manifest line appended LAST, after all real tools:

    - no_tool: Not a tool. Pick this when the message is plain conversation, a
      question you can answer directly, or when none of the listed tools can
      fulfill it.

No server prefix on the line. The JSON contract line is NOT edited — `null`
stays allowed; the pseudo-tool is an addition, not a replacement. Scoring: a
pick of `no_tool` is treated exactly like a null pick (ok iff want is None; on
pos/trap it is a miss like null). schema_ok unchanged (only exact real-tool
matches checked). Presence/gold-rank bookkeeping ignores `no_tool` (it is not
in the registry).

Arms (frozen 278-case eval, prefixed names; sanity `union` rerun first —
acc 0.6763 ± 0.005, neg ≥ 20/24, else STOP):
- `band08nt` — MS-24 band08 selector (band ±0.08, cap 40) + no_tool line.
  Reference (MS-24 scored): acc 0.6619, pos 0.6575, neg 17/24, 1210 tok,
  hit 0.8661, pos|hit 0.7591.
- `uwinthrnt` — MS-23 uwinthr selector (k-means top-2 union ∪ threshold
  windows: window 10, step 7, append rule, band ±0.03) + no_tool line.
  Reference (MS-23 scored): acc 0.6942, pos 0.6850, neg 19/24, 2636 tok,
  hit 0.9567, pos|hit 0.7160.

Bars:
- MS-26a (menu rescue, budget arm): band08nt neg ≥ 20/24 AND acc ≥ union−0.02
  → the budget arm regains admissible null discipline at budget cost. If
  band08nt neg < 20, the menu hypothesis is WRONG for narrow bands and the
  result is archived as a null result.
- MS-26b (adoption): uwinthrnt acc ≥ same-run union acc + 0.02 AND
  neg ≥ 20/24 → ADOPT AS GATE. This is exactly the MS-23a clause uwinthr
  failed by one neg case (19/24), now with the menu fix — a pass retires the
  union gate.
- MS-26c (budget Pareto): band08nt acc ≥ union−0.01 AND tokens ≤ 1585 →
  same bar shape as MS-24b, which band08 missed by one case (0.6619 vs
  0.6627).

PREDICTIONS ON RECORD:
- band08nt neg 19–22 (from 17); uwinthrnt neg 20–23 (from 19). Mechanism:
  narrow bands (1-tool lists, echo traps) become two-option choices where
  abstention has textual weight.
- pos-level effect: −2 to +2 correct on each arm — direction unknown; the
  listed easy-out could siphon borderline pos picks (registered risk).
- Tokens: +30–40 mean per case (one manifest line).

### MS-26 results (2026-10-09, scored)

Run: 3 arms × 278; runner committed pre-data (one canary-caught import fix
committed before any scored record); ledger deleted before the scored run.
Sanity `union` first: 0.6727 (inside ±0.005), neg 21/24, and 0/278
no_tool picks — the control confirms the model never emits `no_tool`
without the extra line.

Disclosure (registered-text vs reference mismatch): the registered
`band08nt` selector text says "band ±0.08, cap 40" — that is MS-24's
**band08cap40** variant, not uncapped band08 (band08 has no cap), while
the attached reference row was band08's. The run implemented the
registered text; measured hit_rate 0.8307 exactly equals band08cap40's.
Correct same-selector references are therefore band08cap40's row below;
verdicts are unchanged under either reference. Registration text kept as
committed; this paragraph is the correction of record.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | no_tool picks | tokens | ms |
|---|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 0/278 | 2265 | 1284 |
| band08nt (vs band08cap40 ref) | 0.6115 | 0.5866 | **21/24** | 0.8307 | 0.7062 | 36/278 | 977 | 746 |
| — reference band08cap40 | 0.6439 | 0.6417 | 16/24 | 0.8307 | 0.7678 | — | 941 | — |
| uwinthrnt (vs uwinthr ref) | 0.6295 | 0.6063 | **21/24** | 0.9567 | 0.6337 | 19/278 | 2672 | 1460 |
| — reference uwinthr | 0.6942 | 0.6850 | 19/24 | 0.9567 | 0.7160 | — | 2636 | — |

Verdicts:
- **MS-26a: NOT RESCUED.** The neg clause PASSED (21/24 ≥ 20; from 16),
  but acc 0.6115 < 0.6527. MS-26c also fails (0.6115 < 0.6627).
- **MS-26b: KEEP UNION.** acc 0.6295 << 0.6927 despite neg 21/24.

Prediction scorecard:
- neg predictions PASSED (registered 19–22 band / 20–23 uwinthr;
  measured 21/24 both).
- pos-effect prediction WRONG, disclosed: registered −2 to +2 correct;
  measured **−14 (band08nt, 163→149) and −16 (uwinthrnt)** on pos-level
  corrects. The registered risk ("listed easy-out could siphon borderline
  pos picks") materialized at ~7× the registered band.

Readings:
- The menu hypothesis is exactly half right, and the wrong half dominates.
  Making abstention a first-class option fixed the narrow-band nulls
  (neg_06 — the get-sum percent question — is the one neg that literally
  picked `no_tool`; total band negs 16→21). But the same line is a
  **false-abstention magnet on real requests**: 54 siphons across arms,
  and they are not borderline — delete_file, git_commit, create_pull_request,
  list_commits. The description clause "a question you can answer directly"
  invites the model to over-trust its own direct-answer ability; LLMs
  always half-believe they can answer directly.
- The whole pos_acc_given_hit crash is the siphon: uwinthr 0.7160→0.6337
  is a drop of 0.082 ≈ 21 siphons/254; one added line out of ~70 menu
  entries costs ~8 points of conditional accuracy. Menu dilution is real
  (bigger uwinthr list: 21 siphons vs band's 33) but dilution softens the
  rescue too.
- The stubborn negs are immune: neg_13 (echo) and neg_03 (add_observations)
  stay wrong with no_tool on the menu — those are genuine confusions, not
  option-availability failures.
- Convergence with MS-25: both planner-side abstention knobs trade
  asymmetrically and badly (intent call: 23/24 nulls for 62 lost
  requests; menu item: +5 nulls for ~19 lost requests). The planner's
  abstention problem is judgment calibration, not option availability —
  the user's original framing ("the agent should be able to decide not
  to use a tool") is vindicated: it could already abstain, and giving
  abstention more salience made it abstain *more*, not *better*.
- Status: gate unchanged (union, 0.6763-class @ 2265 tok, neg 21/24);
  band08nt/uwinthrnt strictly dominated, archived as the menu-framing
  data point. Soft cascade remains named-but-unregistered.

### MS-27 registration: intent as a hint, not a filter — 2026-10-09, pre-data

Question: the cluster filter is measured (top-1 = 63.8% presence — too
lossy; top-2 = 87.8% — shipped), so intent has never been tried as
NON-BINDING context. Does telling the planner what an automatic
classifier guessed — with the menu unchanged — improve picking (per-hit)
without breaking null discipline (neg)? This is the zero-extra-call
version of the soft cascade: the hint comes from the same k=5 space the
gate already computes (top-1 cluster of the query), no LLM intent call.
(An LLM-intent hint would be a separate registration.)

Manipulation (registered verbatim): selector FROZEN at the shipped union
(top-2 clusters, ~70 tools, presence 0.878 — identical shortlists to the
sanity arm). The directive inserts ONE sentence after "Choose a tool for
this request.", before "Available tools:". Two hint strengths:

- `hintname` — top-1 cluster named by its 5 most centroid-central
  members (prefixed names, exactly as they appear in the manifest):
  "Automated pre-classification tags this request as most resembling:
  {n1}, {n2}, {n3}, {n4}, {n5}. The tag is automated and often wrong —
  judge from the full list below."
- `hintfam` — top-1 cluster named only by its distinct server families
  (ordered by member count desc, ties by registry order):
  "Automated pre-classification tags this request as most resembling
  the {servers} tool families. The tag is automated and often wrong —
  judge from the full list below."

The "often wrong" clause is part of the design (anti-anchor mitigation),
not hedging: the tag IS wrong on 36% of should-call cases (top-1
presence 0.638) and always points somewhere on negs.

Arms (frozen 278-case eval; sanity `union` rerun first — acc 0.6763
± 0.005, neg ≥ 20/24, else STOP): `union` (plain directive), `hintname`,
`hintfam`.

Bars:
- MS-27a (adopt): acc ≥ same-run union acc + 0.02 AND neg ≥ 20/24 →
  production directive gains the hint line. Guardrail: neg < 20 rejects
  regardless of acc.
- MS-27b (mechanism, recorded either way): per-hit vs union's 0.7444,
  neg vs 21/24, and PAIRED discordance vs the same-run union arm
  (cases gained / cases lost, same selector).

PREDICTIONS ON RECORD:
- hintname: pos|hit 0.744 → 0.750–0.775 (focus helps when gold is
  in-list); neg 17–21 (anchoring risk on negs, where the tag is noise);
  net acc −4 to +8 cases. tokens +20–40.
- hintfam: pos|hit ±0.01 (too coarse to focus); neg 19–22; acc −2 to
  +4 cases. tokens +20–40.
- Reading rule fixed now: gains concentrated on indirect phrasings with
  gold in-list = focus mechanism; losses concentrated on negs and traps
  = anchor mechanism.

Runner: `tools/mcp_probe/run_hint.py` (imports frozen machinery +
shortlist; writes results_hint.jsonl / summary_hint.json; canary
MCP_HINT_CANARY=1). Committed before any scored run.

### MS-27 results (2026-10-09, scored)

Run: 3 arms × 278; runner committed pre-data; canary clean; ledger
deleted before the scored run. Sanity `union` first: 0.6727 (inside
±0.005), neg 21/24.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 2265 | 1281 |
| hintname | 0.6978 | 0.7087 | **14/24** | 0.8780 | **0.8072** | 2319 | 1376 |
| hintfam | **0.7122** | 0.7205 | **15/24** | 0.8780 | **0.8206** | 2302 | 1349 |

Verdicts:
- **MS-27a: KEEP UNION, both arms** — the FIRST arms in the program to
  clear the accuracy bar (hintname +0.025, hintfam +0.040 vs union;
  hintfam 0.7122 is the highest accuracy ever measured here, above the
  0.7014 gate) — and both tripped the neg guardrail (14/24, 15/24 < 20).

Prediction scorecard:
- pos|hit predictions WRONG LOW, disclosed: hintname registered
  0.750–0.775 → 0.8072; hintfam registered ±0.01 ("too coarse to
  focus") → 0.8206. The comparison INVERTED: the coarse family hint
  beats the specific-name hint on picking.
- neg predictions WRONG LOW: registered 17–21 / 19–22 → 14/24, 15/24.
- tokens +20–40: measured +54/+38 (pass).
- Registered reading rule ("gains concentrated on indirect phrasings"):
  WRONG — paired gains are even (hintname +11 direct/+14 indirect,
  hintfam +10/+12); the tag does not just disambiguate retrieval, it
  primes the right tool-family vocabulary.

Readings (paired vs same-run union, identical shortlists — the
manipulation is one sentence):
- The mechanism split is total and clean: **intent-as-hint is the
  strongest picking aid ever measured** (per-hit 0.744 → 0.807/0.821 —
  conditional accuracy ABOVE the flat-121 level, i.e. ~70 tools with a
  one-line tag pick better than all 121 untagged) **and the strongest
  null-breaker ever measured** (neg 21 → 14/15; 8 and 6 negs flip to
  wrong picks — neg_06 get-sum and neg_10 flip in both arms). Gains +26
  /−19 (hintname), +22/−11 (hintfam) — the anchor tax lands exactly on
  negs, the focus gain exactly on should-call.
- The inversion (families > names) says the hint works by orienting,
  not by pointing: naming five tools steals picks even when the tag is
  right-ish but the gold is elsewhere; naming families orients without
  stealing.
- Composition is the obvious next step and is now registered (MS-28):
  MS-25's blind intent call caught 23/24 negs — suppress the hint
  exactly where its damage lands (intent says CHAT → plain directive,
  never an early exit), keep it where its gain lands (TOOL → tagged
  directive). Failure modes of MS-25 and MS-27 are complementary by
  construction.
- Status: gate unchanged; hintfam archived as the accuracy record
  (0.7122) that cannot ship (neg 15/24); hint data point: one sentence
  ≈ +40 tokens swings per-hit +8 pts and negs −7.

### MS-28 registration: conditional hint (soft cascade) — 2026-10-09, pre-data

Question: MS-25 and MS-27 measured complementary failure modes. The blind
intent call (frozen MS-25 wording, temp 0) catches 23/24 nulls but a hard
exit rejects 62/254 real requests. The hintfam tag lifts per-hit 0.744 →
0.821 (program-record acc 0.7122) but breaks nulls (15/24). Composition:
run the intent call, then let it CONTROL ONLY THE HINT — CHAT verdict →
planner sees the plain union directive (menu intact, no early exit,
never rejected); TOOL verdict → planner sees the union menu PLUS the
registered MS-27 hintfam sentence. The hint's damage lands exactly on
intent-CHAT cases (mostly negs) and its gain exactly on intent-TOOL
should-call cases. Zero removal anywhere; the intent call only chooses
between two full menus.

Manipulation (registered verbatim): selector FROZEN at union (top-2
clusters) for both branches. Hint branch = MS-27 `hintfam` sentence and
clause, byte-identical (top-1 cluster server families + "often wrong"
tail), inserted after "Choose a tool for this request." Intent branch =
MS-25's blind INTENT_DIRECTIVE call (no manifest, no tool names), temp
0, max_tokens 4, decision = text starts with TOOL else CHAT.

Deployment cost note (registered so adoption is priced honestly):
adopting means the production gate gains one extra LLM call per
utterance (~150–400 prompt tokens, sub-second on the local server) and
a branch on its verdict — beyond the +38-token hint line itself.

Arms (frozen 278-case eval; sanity `union` rerun first — acc 0.6763
± 0.005, neg ≥ 20/24, else STOP): `union` (plain), `softhint` (the
conditional above; hintfam conditioning chosen because MS-27 measured
it ≥ hintname on both acc and neg — hintname is not rerun).

Bars:
- MS-28a (adopt): acc ≥ same-run union acc + 0.02 AND neg ≥ 20/24 →
  the gate gains the conditional hint (intent call + branch). Guardrail:
  neg < 20 rejects regardless of acc.
- MS-28b (mechanism, recorded either way): negs split by intent verdict
  (how many TOOL-flagged negs the hint still breaks); paired gains vs
  union on should-call (incl. how many of MS-25's 62 false-CHAT cases
  the plain menu converts); tokens.

PREDICTIONS ON RECORD:
- Intent call reproduces MS-25: ~50–75/254 should-call flagged CHAT.
- neg 20–23: ~20 negs flagged CHAT (hint suppressed → union-level
  nulls); of the ~2–4 TOOL-flagged negs the hint breaks −1 to −2.
- acc 0.69–0.71 (hint gain on the ~180–200 flagged-TOOL should-call,
  flagged-CHAT cases keep union behavior ±3).
- tokens mean 2350–2600 (intent call added on every case; planner
  skipped for no one — the soft part).
- Disclosure rules: should-call CHAT-flags outside 50–75 = serving
  drift note; neg < 20 = KEEP UNION regardless of acc.

Runner: `tools/mcp_probe/run_softhint.py` (imports frozen machinery,
intent from run_step1, tag/hint_directive from run_hint; writes
results_softhint.jsonl / summary_softhint.json; canary
MCP_SOFTHINT_CANARY=1). Committed before any scored run.

### MS-28 results (2026-10-09, scored)

Run: 2 arms × 278; runner committed pre-data; canary clean; ledger
deleted before the scored run. Sanity `union` first: 0.6727 (inside
±0.005), neg 21/24 — fourth consecutive exact reproduction.

| arm | acc | pos_acc | neg | hit_rate | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 2265 | 1281 |
| softhint | **0.7194** | 0.7047 | **21/24** | 0.8780 | 0.8027 | 2405 | 1307 |

Verdicts:
- **MS-28a: ADOPT CONDITIONAL HINT.** acc 0.7194 ≥ 0.6927 (union+0.02)
  AND neg 21/24 ≥ 20. First adoption verdict of the gate-search program:
  +13 correct cases vs the shipped gate (200 vs 187) at equal null
  discipline, +140 mean tokens, +26 ms (the intent call is nearly free
  on the local server). 0.7194 is the highest accuracy ever measured on
  this eval, above the unprefixed gate reference 0.7014 and the
  unconditional hintfam record 0.7122.
- MS-28b (mechanism): intent flags reproduced MS-25 exactly —
  62/254 should-call CHAT (bit-level same as the frozen run), 20/24 neg
  CHAT. The hint's anchor damage was confined to its true domain: of
  the 4 TOOL-flagged negs, the hint broke 1; the 20 CHAT-flagged negs
  null like union (21/24 total, = union). The intent call's false-CHAT
  weakness was demoted from fatal (62 hard exits in MS-25) to mild:
  the plain union menu still converted 36/62 of those cases. Paired:
  gained 16 / lost 3. pos|hit 0.744 → 0.803.

Prediction scorecard: intent-flag band (50–75) PASS (62); neg band
(20–23) PASS (21); token band (2350–2600) PASS (2405); acc band
(0.69–0.71) WRONG HIGH, disclosed — measured 0.7194, ~1 pt above the
registered ceiling. Direction of every mechanism prediction held.

Reading:
- The composition worked exactly as designed: two failure modes bought
  each other out. MS-25's hard gate died on false CHAT (62 rejections);
  in the soft cascade a false CHAT only withholds the hint (menu still
  serves, 36/62 converted). MS-27's hint died on neg anchoring (−7
  nulls); the intent call suppresses the hint on 20/24 negs, so the
  anchor tax is prepaid where it matters. Neither half is shippable
  alone (MS-25 acc 0.547, MS-27 neg 15/24); together they dominate the
  shipped gate on accuracy at equal nulls.
- The hint itself remains the strongest per-case picking lever known
  here (pos|hit +6 pts for ~38 tokens); the intent call is the
  cheapest working guard for it.
- Deployment shape (for the production integration step): in
  tool_flow, run the MS-25 intent call on the utterance; CHAT → gate
  directive as today; TOOL → gate directive + one sentence naming the
  top-1 cluster's server families from the existing index, with the
  registered "often wrong" clause. Config surface stays untouched.
- Status: measured design adopted at the protocol level; production
  wiring is a separate implementation step and NOT part of this
  experiment.

### MS-29 registration: window-union recall + intent-gated hint — 2026-10-09, pre-data

Rationale: the softhint error budget is 31 retrieval misses (presence
0.878) / 44 wrong picks / 3 negs. MS-23's uwinthr selector (k-means
top-2 union ∪ threshold windows: WIN=10, STEP=7, append rule, band
±0.03 of the best window) measured presence 0.9567 — recovering ~20 of
the 31 misses — and was rejected ONLY on neg 19/24, one short of the
guardrail. The adopted design no longer asks the selector to defend
nulls: MS-28 measured the intent call confining hint damage to 1 neg.
This experiment stacks the three best-measured ingredients on orthogonal
axes: uwinthr recall + hintfam picking + intent-guarded nulls.

Manipulation (registered verbatim): identical to MS-28 softhint (blind
MS-25 intent call temp 0; CHAT → plain directive, never an early exit;
TOOL → directive + MS-27 hintfam sentence with "often wrong" clause;
tag = top-1 cluster server families from the same k=5 space) with ONE
change: the shortlist on BOTH branches is uwinthr instead of union.
The sanity arm stays the shipped union gate, plain directive (must
reproduce 0.6763 ± 0.005, neg ≥ 20/24, else STOP).

References (frozen scored runs, same eval):
- uwinthr plain (MS-23): acc 0.6942, neg 19/24, presence 0.9567,
  pos|hit 0.7160, 2636 tok.
- softhint (MS-28, incumbent): acc 0.7194, neg 21/24, presence 0.8780,
  pos|hit 0.8027, 2405 tok.

Arms: `union` (sanity), `winhint` (the conditional hint on the uwinthr
selector).

Bars:
- MS-29a (beat gate): acc ≥ same-run union acc + 0.02 AND neg ≥ 20/24.
- MS-29b (beat incumbent, become the wiring target): acc ≥ 0.7194 AND
  neg ≥ 20/24. Priced cost: windows add ~350–400 mean tokens vs
  softhint (expected ~2700–2900); the wiring target is whichever of
  {softhint, winhint} passes with the higher acc.

PREDICTIONS ON RECORD:
- presence 0.950–0.960 (four consecutive exact pre-check→scored
  transfers; drift outside ±0.005 = machinery note, no verdict).
- acc 0.74–0.77 (242–244 hits × 0.78–0.81 + neg 20–22).
- neg 19–22: CHAT-flagged negs (~20) null at plain-uwinthr level;
  TOOL-flagged (~2–4) lose −0 to −2 to the hint.
- pos|hit 0.78–0.82 (hint measured 0.803 on the union list; the bigger
  uwinthr list may dilute).
- tokens 2650–2900.
- Disclosure: neg < 20 → not wireable regardless of acc; acc below
  softhint's 0.7194 → softhint remains the wiring target (windows not
  justified).

Runner: `tools/mcp_probe/run_winhint.py` (imports frozen machinery;
shortlist from run_beam; windows per MS-22/23 registered tiling; intent
from run_step1; tag/hint from run_hint; writes results_winhint.jsonl /
summary_winhint.json; canary MCP_WINHINT_CANARY=1). Committed before
any scored run.

### MS-29 results (2026-10-09, scored)

Run: 2 arms × 278; runner committed pre-data; canary clean; ledger
deleted before the scored run. Sanity `union` first: 0.6727 (inside
±0.005), neg 21/24 — fifth consecutive exact reproduction.

| arm | acc | pos_acc | neg | presence | pos\|hit | tokens | ms |
|---|---|---|---|---|---|---|---|
| union (sanity) | 0.6727 | 0.6535 | 21/24 | 0.8780 | 0.7444 | 2265 | 1285 |
| winhint | **0.7338** | 0.7244 | **20/24** | **0.9567** | 0.7572 | 2776 | 1513 |

Verdicts:
- **MS-29a: BEATS GATE** (0.7338 ≥ 0.6927, neg 20/24 ≥ 20).
- **MS-29b: WIRING TARGET** — acc 0.7338 > softhint's 0.7194 (+4
  cases, 204 vs 200; +17 vs the shipped gate's 187) at neg 20 ≥ 20.
  Per the registered rule, winhint replaces softhint as the production
  wiring design.

Prediction scorecard: presence PASS (0.9567, exact transfer, fifth
consecutive); neg PASS (20, in 19–22); tokens PASS (2776, in
2650–2900); acc WRONG, disclosed — registered 0.74–0.77, measured
0.7338, ~2 cases under the floor; pos|hit WRONG, disclosed — registered
0.78–0.82, measured 0.7572.

Readings:
- The recall bet paid exactly as projected (+20 recovered hits,
  presence 0.878 → 0.9567), but the hint's picking gain dilutes on the
  bigger list: +8 pts per-hit on the union menu (MS-27) became +4 pts
  here (0.716 → 0.757 vs plain-uwinthr). Mechanism: window-added tools
  are noise the families-tag doesn't cover, so the hint steers less
  per pick — the recall windfall still dominates (+4 net cases over
  softhint; paired vs union: gained 39 / lost 22).
- Nulls held at the guardrail edge: 20/24 (union: 21). Intent flags:
  should-call CHAT 66/254 (MS-25/28 measured 62 — small serving
  variance, inside the registered 50–75 band; the intent call is not
  bit-deterministic across loads), neg CHAT 20/24; the hint broke 1 of
  4 TOOL-flagged negs, the remaining misses are uwinthr's own
  base-level null behavior.
- Cost of the record: 2776 tok (+511 vs shipped gate, +371 vs
  softhint), 1513 ms (+228). And zero margin on the guardrail: neg 20
  is exactly at the bar — one case of serving noise below 20 would
  flip the wiring decision back to softhint (0.7194, neg 21).
- Program trajectory on this eval: flat 0.6835 → shipped gate 0.6727
  (prefixed) → softhint 0.7194 → winhint 0.7338 — the adopted design
  now beats showing every tool by 5 points at 74% of flat's tokens.
- Status: winhint registered as wiring target; production integration
  (intent call + hint branch + window-union shortlist in
  web/server.py's tool_flow, using the existing index) is the next
  implementation step, separate from this experiment.
