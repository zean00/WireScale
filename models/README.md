# Tracked CLM head artifacts

The production frozen-encoder readout stack, tracked here so the ledger is
self-contained. Copied byte-identical (sha256-verified) from the WireSense
repo, where they were first committed (`8a8e59e`, "track the decision-tree
heads/refs in-repo"); trained in the rewire program (entry U, seed
2026-10-03). Production loaders: WireSense `web/server.py` (router walk),
`tools/tree_probe.py`, `tools/eval_router.py`.

| file | size | what |
|---|---|---|
| `arm_C_heads.npz` | 29 MB | the CLM head: trunk `fc1 (2560→2048)` + LayerNorm + `fc2 (2048→1024)`, config embedded as `cfg_json` = `{"dim": 2560, "width": 2048, "projection_dim": 1024, "activation": "gelu", "layernorm": true, "residual": true, "seed": 20261003, "post_train_pairs": 504}` |
| `router_embeddings.npz` | 2 MB | per-node reference banks (the readout layer): L2-normalized embedding refs per router node (`root.*`, `think.*`, `feedback.*`, `backchannel`) |
| `tools/decision_tree.json` | 13 KB | the tree the readouts walk: node structure + text pools per node |

Note: this lab's own trunk (`heads.py`) is the same lineage but not
weight-identical — it drops the residual and adds the arm-C conditioning
variants (parent one-hot concat / FiLM). The production artifact above is
self-describing via `cfg_json`.

**Model-bound (mirrors the WireSense disclaimer).** These weights and refs
are valid ONLY against the encoder they were recorded from: Qwen3.5-4B
UD-Q4_K_XL served with `--embedding --pooling last` (2560-dim readouts).
Any change of LLM, quantization, or pooling invalidates them silently —
re-record, don't reuse.
