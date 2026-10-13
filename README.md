# WireScale

*(formerly `tree-lab`)*

A pre-registered laboratory for decision structure. One frozen encoder, many
readouts — and one rule: **register the prediction before the data, then obey
the verdict.** Three scored experiments live here, plus one registered and
awaiting its data:

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
4. **Chunk relevance** (R-series) — *registered 2026-10-13 pre-data, no
   scored run yet*: within-document chunk selection. Does pointwise
   (question, chunk) joint judgment beat bi-embedding cosine when the
   question does not share the chunk's vocabulary? Registered in
   [protocol.md](protocol.md); runner `chunkrel.py`; question files must be
   hand-audited and frozen before the scored run.

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

**0 of 4 headline predictions passed as registered. The failures are the
finding.**

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
| `chunkrel.py` | R-series runner (chunk relevance: bi-embed cosine V vs pointwise judge J-llm vs pair head J-combo vs listwise J-menu) — **registered pre-data, no scored run yet** |
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
.venv/bin/python adjudicate.py          # recompute P1–P4 verdicts from results/
```

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
