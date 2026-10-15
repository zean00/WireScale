# WireSense hand-off: what the retrieval program transfers

*From WireScale (`tree-lab`), 2026-10-15. Every number below is a committed
artifact — see `protocol.md` and `results/*.json` in the lab repo. This note
is exported verbatim into WireSense; the reverse direction already happened
once (`PROTOCOL-mcp-selection.md` came the other way at `876e1e8`).*

**The rule of this note:** the lab's *numbers* are model-bound and
corpus-bound (frozen Qwen3.5-4B + Qwen3-Embedding-0.6B, 58 questions over a
180-page pinned corpus). The *recipes and rules* transfer. Re-register any
number on WireSense's own corpus before trusting it — register-first applies
in the destination, not just the source.

---

## T1 — The shortlist-and-verify reader (C-series + D-series)

For any "answer from documentation" path (manuals, policy pages, user setup),
the scored shape is:

1. Rank pages with the dedicated embedder (plain query verbatim — the model
   card's instruct prefix *hurt*, E-series EB).
2. Judge the shortlist **one page at a time in rank order**; stop at the
   first page whose calibrated p_yes crosses the bar.
3. Answer from that single page only.
4. If nothing crosses, fall back to rank-1.

Lab numbers to re-register against: shortlist-of-8 at bar τ=.70 scored
**.448 answers-correct at 1,423 tok/q** — ahead of the best top-k RAG form
(.397) and the full read-the-document machine (.431 @ 5,428). Plain top-k RAG
flatlines at .33–.40 for every k because synthesis collapses as context grows
(.77 → .43 from k=1 to 8) while context hit climbs — **stuffing pages is the
anti-pattern this replaces** (B-series).

Deployment notes:

- Keep the shortlist generous (~8). Mean pages judged is ~1.6 at the strict
  bar; insurance against gold-beyond-the-shortlist is nearly free. The full
  cost of shrinking to 4 is measured (ASM-4: −3.5 pts) and it lives in
  fallback quality, not in hits.
- The rank-1 fallback is not a failure mode to engineer away: its quality
  *rises* with the bar (fallback hit .069 → .224 across τ=.30–.80).

## T2 — Stiffen the bar; never re-sample the gate (D-series)

The single most transferable rule of the season, for **every yes/no gate** in
the loop (tool-gate confirmations, MS-29's intent-gated hint, routing or
hallucination checks):

- Verifier judges fail **confidently**, not noisily. Adding self-consistency
  sampling (3× temp-1.0) changed **zero** stop decisions on 58/58 questions —
  3× the judge tokens for exactly nothing.
- The profitable dial is the **decision bar**. The judge's misses are
  confident rejections (median cal-p ≈ .03); its costly failure is
  confident-wrong *approvals*. Raising the bar converts those into
  recoverable fallbacks and was worth +3.5 pts at 26% of the cost.
- Calibrate first, then place the bar: one temperature (T=0.66, Z-series
  method — fit on a split, ECE .105 → .081), then pick the bar on the gate's
  own trade-off curve. The lab's sweep: capture flat for τ .30–.70,
  collapsing at .80; the sweet spot was the strict side of the plateau.
- Same culture as the standing "score paths by **mean, never sum**" rule:
  one measured instinct-correction, cheap to apply, expensive to skip.

## T3 — Audit the grader before tuning to it (G-series + S-amendment)

Before letting any LLM-graded eval metric steer WireSense development
(including tuning to MS-29's .7338-vs-.6727 gap):

- **The ritual:** re-grade the *same committed answers* under an altered
  grader prompt (rewording suffices) and under a sampled faithful-prompt
  grader. Lab result: orderings survived both (GA PASS ×2); margins did not —
  the reworded grader rescued pure RAG +.24 and tied the machine's headline
  margin.
- **The rule:** trust orderings, never margins. If a proposed change's edge
  doesn't survive a grader-prompt perturbation, it isn't an edge.
- **The sizing rule:** at n=58, *every* answer-level margin sat inside
  statistical noise (exact McNemar + paired bootstrap, six comparisons,
  p .125–1.0, direction unanimous). A +5-point claim needs an eval set sized
  to make it a claim. WireSense's eval sets should be grown with this in
  mind before margins are tuned to.

## T4 — Standing transfers (already live)

- **Beam-2 everywhere** (trees, routing, graphs — four replications, largest
  margins in the lab): keep two candidates alive at any coarse choice.
- **Mean, never sum** for path scores of different depth (the 13-point crash).
- The MS-series record itself (`PROTOCOL-mcp-selection.md`, MS-1..MS-29):
  MS-29 winhint remains the registered wiring target, pending integration.

## T5 — Measured anti-lessons (do not do these)

Each paid for once; don't pay again:

- **Don't rewrite the user's query** into doc-speak (−6.9 indirect; Y-series).
- **Don't add the embedder's official instruct prefix** (−3.5; E-series EB).
- **Don't stuff context** — more excerpts ≠ better answers (B-series).
- **Don't commit early to a single route** — greedy walks lost six straight
  verdicts; 3-hop route lookahead lost worse (−25; K-series Amendment 2).
- **Don't sum variable-depth path scores** (the hidden 13-point crash).

## Wiring this into the pending MS-29 integration

1. Keep the window-union shortlist; order it with the current ranker.
2. Gate in rank order with a calibrated bar (T2): fit T on a held-out split
   of WireSense queries, start on the strict side (the lab's winning bar was
   .70 on a .5-defaulted plateau).
3. On no-cross, fall back to the shipped-gate behavior — the fallback is the
   safe path, and its quality improves with the strict bar.
4. Before declaring the wiring won, run the T3 ritual on the eval answers:
   confirm the winhint-over-shipped-gate *ordering* survives an altered
   grader prompt, and size the eval set so the margin is a claim.

## Proposed next series in the lab (proposals, not yet registered)

- **#4 Doc-selection** — the layer above T1: *which* document/collection
  gets read. Arms: embedder doc-rank + within-doc machine vs flat scan vs
  doc-order walk. The K/W precedent says flat wins when it fits; the open
  question is the budget-matched crossover.
- **#5 Abstention** — replace T1's rank-1 fallback with abstention
  ("I don't know") when the pool's max calibrated p sits below a floor.
  Bars should be risk-relaxed: accuracy-given-answered ≥ the rank-1 arm's,
  with coverage reported as a first-class number. Directly wired to what a
  voice assistant should say when nothing verifies.

---

*Evidence index (lab repo): `asm.py`/`assembly.json` (A), `topk.py`/
`topk.json` (B), `poolstop.py`/`poolstop.json` (C), `stats.py`/
`results/stats_amendment.json` (S), `pooljudge.py`/`results/pooljudge.json`
(D), `regrade.py`/`results/regrade.json` (G); `protocol.md` is the full
decision record. Register first, measure, admit when wrong.*
