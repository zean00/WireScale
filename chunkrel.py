"""R-series: chunk-relevance readouts (protocol.md, registered pre-data).

Within-document chunk selection: does pointwise (question, chunk) joint
judgment (J-llm) beat bi-embedding cosine (V) when the question does not
share the chunk's vocabulary (the indirect recipe = the markedness dial)?
J-combo is the cacheable frozen pair head; J-menu the listwise control.
Exhaustive per-document — no pruning structure (P/S-series verdict).

    python chunkrel.py --smoke            # plumbing check, never scored
    python chunkrel.py --gen-questions    # write freeze candidates (AUDIT)
    python chunkrel.py                    # scored run (needs frozen files)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn

import embed_client as EC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
QFILE = os.path.join(RES, "r_questions.json")

DOCS = ["README.md", "protocol.md", "PROTOCOL-mcp-selection.md",
        "models/README.md"]
NEG_DOC = "README.md"            # hand negatives judged here (registered)
CHUNK_CHARS, OVERLAP = 1100, 150
SPLIT_SEED, MENU_N, MENU_PREVIEW, SEED_MENU = 11, 20, 200, 999
GEN_TEMP, GEN_SEED, LLM_MAXT = 0.8, 7, 16
COMBO_SEED, PAIR_SEED = 0, 31

HAND_NEGS = [
    "What is seventeen times twenty-three?",
    "Can you tell me a joke?",
    "What's the weather like in Jakarta right now?",
    "Who won the most recent World Cup?",
    "Remind me to water the plants tomorrow morning.",
    "What is the capital of Australia?",
    "Thanks, that's everything for today.",
    "How do I bake sourdough bread at home?",
]

JUDGE_SYS = ("You are a strict relevance judge. Answer with exactly one "
             "word: yes or no.")
MENU_SYS = ("You pick the passage most likely to help answer a question. "
            "Reply with ONLY the number of the passage, or 0 if none of "
            "them help.")
GEN_SYS = "Output only the question itself. No preamble, no answer, no quotes."
RECIPES = {
    "direct": ("Write one question of at most 20 words whose answer is "
               "contained in this passage. Reuse the passage's own "
               "vocabulary where natural.\n\nPASSAGE:\n{chunk}"),
    "indirect": ("Write one question of at most 20 words that a person with "
                 "a concrete need would ask when this passage is exactly "
                 "what they need to read. Do NOT reuse the passage's "
                 "vocabulary; describe the underlying need or goal, not the "
                 "passage's content.\n\nPASSAGE:\n{chunk}"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- chunks

def chunk_doc(text):
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        if end < n:
            snap = text.rfind(" ", end - 300, end)
            if snap > start + 200:
                end = snap
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - OVERLAP, start + 1)
    return out


def build_chunkmap():
    cm = {}
    for d in DOCS:
        with open(os.path.join(HERE, d), encoding="utf-8") as f:
            ch = chunk_doc(f.read())
        if len(ch) >= 2:
            cm[d] = ch
        else:
            log(f"  dropping {d} (<2 chunks)")
    return cm


def split_indices(n_chunks):
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_chunks)
    half = min(12, max(2, n_chunks // 2))
    tr = sorted(int(i) for i in perm[:half])
    ev = sorted(int(i) for i in perm[half:2 * half])
    if not ev and len(tr) >= 2:
        ev, tr = tr[-1:], tr[:-1]
    return tr, ev


# ------------------------------------------------------------------- llm

def llm_chat(messages, temp, max_tokens, seed=None, logprobs=False,
             retries=2):
    payload = {"messages": messages, "temperature": temp,
               "max_tokens": max_tokens, "id_slot": EC.SLOT,
               "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        payload["seed"] = seed
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 4
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                EC.BASE + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise RuntimeError(f"llm_chat failed after retries: {last}")


def p_yes_of(resp):
    """Normalized first-token yes/no logprob mass; None if unserved."""
    try:
        tl = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    y = n = 0.0
    for t in tl:
        tok = (t.get("token") or "").strip().lower()
        if tok.startswith("y"):
            y += float(np.exp(t["logprob"]))
        elif tok.startswith("n"):
            n += float(np.exp(t["logprob"]))
    if y + n <= 0:
        return None
    return y / (y + n)


def judge(q, chunk):
    r = llm_chat([{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content":
                   f"Question: {q}\n\nPassage:\n{chunk}\n\n"
                   "Does the passage contain information that helps answer "
                   "the question?"}],
                  temp=0, max_tokens=LLM_MAXT, logprobs=True)
    content = r["choices"][0]["message"]["content"].strip().lower()
    p = p_yes_of(r)
    yes = (p >= 0.5) if p is not None else \
        bool(content) and content.split()[0].startswith("y")
    return yes, p, r.get("usage", {})


def menu_subset(doc, gold, chunkmap):
    ch = chunkmap[doc]
    rng = np.random.default_rng(SEED_MENU + gold)
    if len(ch) <= MENU_N:
        idxs = list(range(len(ch)))
    else:
        rest = [i for i in range(len(ch)) if i != gold]
        idxs = [gold] + list(rng.choice(rest, MENU_N - 1, replace=False))
        rng.shuffle(idxs)
    prev = [ch[i][:MENU_PREVIEW].replace("\n", " ") for i in idxs]
    return idxs, prev


def menu_pick(q, previews):
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(previews))
    r = llm_chat([{"role": "system", "content": MENU_SYS},
                  {"role": "user", "content":
                   f"Question: {q}\n\nPassages:\n{numbered}\n\n"
                   "Reply with only the number (0 if none)."}],
                  temp=0, max_tokens=LLM_MAXT)
    txt = r["choices"][0]["message"]["content"].strip()
    for tok in txt.replace(".", " ").replace(",", " ").split():
        if tok.isdigit():
            k = int(tok)
            if 0 <= k <= len(previews):
                return k, r.get("usage", {})
    return 0, r.get("usage", {})


# ------------------------------------------------------------- questions

def gen_questions(chunkmap):
    out = {"train": [], "eval": []}
    i = 0
    for doc in DOCS:
        if doc not in chunkmap:
            continue
        tr, ev = split_indices(len(chunkmap[doc]))
        for split, idxs in (("train", tr), ("eval", ev)):
            for ci in idxs:
                for recipe, tmpl in RECIPES.items():
                    r = llm_chat(
                        [{"role": "system", "content": GEN_SYS},
                         {"role": "user", "content":
                          tmpl.format(chunk=chunkmap[doc][ci])}],
                        temp=GEN_TEMP, max_tokens=64,
                        seed=GEN_SEED * 100003 + 7 * i)
                    i += 1
                    q = r["choices"][0]["message"]["content"].strip().strip('"')
                    out[split].append({"doc": doc, "chunk": ci,
                                       "recipe": recipe, "q": q})
                    log(f"  {doc}[{ci}] {recipe}: {q[:70]}")
    out["hand_negs"] = HAND_NEGS
    out["commit"] = git_commit()
    out["model_fp"] = EC.fingerprint()
    return out


# ------------------------------------------------------------ embeddings

def vec(cache, t):
    v = cache[t].astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def embed_all(chunkmap, texts, cache):
    everything = [t for ch in chunkmap.values() for t in ch] + list(texts)
    EC.embed(everything, cache)
    EC.save_cache(cache)
    return {d: np.stack([vec(cache, t) for t in ch])
            for d, ch in chunkmap.items()}


# ---------------------------------------------------------------- combo

def feats(qv, cv):
    return np.concatenate([qv, cv, np.abs(qv - cv), qv * cv]).astype(np.float32)


class ComboNet(nn.Module):
    def __init__(self, d=4 * 2560):
        super().__init__()
        self.fc1 = nn.Linear(d, 2048)
        self.act = nn.GELU(approximate="tanh")
        self.ln = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        return self.head(self.fc2(self.ln(self.act(self.fc1(x))))).squeeze(-1)


def train_combo(pairs, cache, seed=COMBO_SEED, epochs=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    X = np.stack([feats(vec(cache, q), vec(cache, c)) for q, c, _, _, _, _
                  in pairs])
    y = np.array([l for _, _, l, _, _, _ in pairs], dtype=np.float32)
    val = np.zeros(len(y), dtype=bool)
    for cls in (0.0, 1.0):
        idx = np.nonzero(y == cls)[0]
        if len(idx):
            val[rng.choice(idx, size=max(1, len(idx) // 10),
                           replace=False)] = True
    tr_idx = np.nonzero(~val)[0]
    net = ComboNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = bce(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(bce(net(Xt[val]), yt[val]))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone()
                          for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def build_pairs(qdata, chunkmap, cache):
    """(q, chunk_text, label, doc, chunk_idx, entry) — train split only."""
    rng = np.random.default_rng(PAIR_SEED)
    docs = [d for d in chunkmap]
    pairs = []
    for e in qdata["train"]:
        ch = chunkmap[e["doc"]]
        gold = e["chunk"]
        pairs.append((e["q"], ch[gold], 1, e["doc"], gold, e))
        others = [i for i in range(len(ch)) if i != gold]
        if others:
            for i in rng.choice(others, size=min(2, len(others)),
                                replace=False):
                pairs.append((e["q"], ch[int(i)], 0, e["doc"], int(i), e))
        wd = [d for d in docs if d != e["doc"]]
        if wd:
            wdoc = wd[int(rng.integers(len(wd)))]
            wchunk = int(rng.integers(len(chunkmap[wdoc])))
            pairs.append((e["q"], chunkmap[wdoc][wchunk], 0, wdoc, wchunk, e))
    return pairs


# -------------------------------------------------------------- metrics

def tune_tau(cos_gold, cos_neg):
    vals = sorted(set(np.concatenate([cos_gold, cos_neg]).tolist()))
    best_tau, best_acc = 0.0, -1.0
    for tau in vals:
        acc = ((cos_gold >= tau).mean() + (cos_neg < tau).mean()) / 2
        if acc > best_acc:
            best_tau, best_acc = float(tau), float(acc)
    return best_tau, best_acc


def ece(ps, ys, bins=10):
    if len(ps) < 20:
        return None
    ps, ys = np.asarray(ps, float), np.asarray(ys, float)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (ps >= lo) & ((ps < hi) if b < bins - 1 else (ps <= 1.0))
        if m.any():
            e += float(m.mean()) * abs(ps[m].mean() - ys[m].mean())
    return e


def agg(records, recipe, key):
    vals = [r[key] for r in records if r["recipe"] == recipe]
    return float(np.mean(vals)) if vals else None


# ------------------------------------------------------------ scored run

def run_scored():
    if not os.path.exists(QFILE):
        sys.exit("results/r_questions.json missing: run --gen-questions, "
                 "HAND-AUDIT the questions, then freeze (protocol R-2).")
    with open(QFILE) as f:
        qdata = json.load(f)
    chunkmap = build_chunkmap()
    texts = ([e["q"] for s in ("train", "eval") for e in qdata[s]]
             + qdata["hand_negs"])
    cache = EC.load_cache()
    mats = embed_all(chunkmap, texts, cache)
    log(f"docs: { {d: len(c) for d, c in chunkmap.items()} }")

    tpairs = build_pairs(qdata, chunkmap, cache)
    cg = np.array([float(vec(cache, q) @ vec(cache, c))
                   for q, c, l, _, _, _ in tpairs if l == 1])
    cn = np.array([float(vec(cache, q) @ vec(cache, c))
                   for q, c, l, _, _, _ in tpairs if l == 0])
    tau, tau_acc = tune_tau(cg, cn)
    log(f"V tau={tau:.4f} (train pair acc {tau_acc:.3f}, "
        f"{len(tpairs)} pairs)")

    net, val_ce = train_combo(tpairs, cache)
    log(f"J-combo trained valBCE={val_ce:.4f}")

    records, ece_c, ece_j = [], [], []
    usage = {"J-llm": [0, 0, 0.0, 0], "J-menu": [0, 0, 0.0, 0]}
    p_cover = [0, 0]
    for qi, e in enumerate(qdata["eval"]):
        ch, M, gold, qv = (chunkmap[e["doc"]], mats[e["doc"]], e["chunk"],
                           vec(cache, e["q"]))
        order = np.argsort(-(M @ qv))
        v_p1, v_p3 = int(order[0] == gold), int(gold in order[:3])
        with torch.no_grad():
            F = np.stack([feats(qv, M[i]) for i in range(len(ch))])
            p = torch.sigmoid(net(torch.from_numpy(F))).numpy()
        corder = np.argsort(-p)
        c_p1, c_p3 = int(corder[0] == gold), int(gold in corder[:3])
        ece_c += [(float(p[i]), float(i == gold)) for i in range(len(ch))]
        t0 = time.time()
        yes_set = []
        for i, ctext in enumerate(ch):
            yes, pj, u = judge(e["q"], ctext)
            if pj is not None:
                p_cover[0] += 1
                ece_j.append((pj, float(i == gold)))
            p_cover[1] += 1
            usage["J-llm"][0] += u.get("prompt_tokens", 0)
            usage["J-llm"][1] += u.get("completion_tokens", 0)
            if yes:
                yes_set.append(i)
        usage["J-llm"][2] += (time.time() - t0) * 1000
        usage["J-llm"][3] += 1
        idxs, prev = menu_subset(e["doc"], gold, chunkmap)
        t0 = time.time()
        k, u = menu_pick(e["q"], prev)
        usage["J-menu"][0] += u.get("prompt_tokens", 0)
        usage["J-menu"][1] += u.get("completion_tokens", 0)
        usage["J-menu"][2] += (time.time() - t0) * 1000
        usage["J-menu"][3] += 1
        menu_hit = int(k >= 1 and idxs[k - 1] == gold)
        records.append({"doc": e["doc"], "chunk": gold,
                        "recipe": e["recipe"], "q": e["q"],
                        "v_p1": v_p1, "v_p3": v_p3,
                        "c_p1": c_p1, "c_p3": c_p3,
                        "jllm_recall": int(gold in yes_set),
                        "jllm_nyes": len(yes_set),
                        "menu_hit": menu_hit})
        log(f"  q{qi + 1}/{len(qdata['eval'])} {e['recipe']:8s} "
            f"v3={v_p3} c3={c_p3} jllm={gold in yes_set} "
            f"nyes={len(yes_set)} menu={menu_hit}")

    negs = []
    chn, Mn = chunkmap[NEG_DOC], mats[NEG_DOC]
    for q in qdata["hand_negs"]:
        qv = vec(cache, q)
        v_abst = int(float((Mn @ qv).max()) < tau)
        with torch.no_grad():
            Fn = np.stack([feats(qv, Mn[i]) for i in range(len(chn))])
            pn = torch.sigmoid(net(torch.from_numpy(Fn))).numpy()
        c_abst = int(float(pn.max()) < 0.5)
        j_abst = int(not any(judge(q, c)[0] for c in chn))
        idxs, prev = menu_subset(NEG_DOC, 0, chunkmap)
        m_abst = int(menu_pick(q, prev)[0] == 0)
        negs.append({"q": q, "V": v_abst, "J-combo": c_abst,
                     "J-llm": j_abst, "J-menu": m_abst})

    def rp(recipe):
        return {"v_p1": agg(records, recipe, "v_p1"),
                "v_p3": agg(records, recipe, "v_p3"),
                "c_p1": agg(records, recipe, "c_p1"),
                "c_p3": agg(records, recipe, "c_p3"),
                "jllm_recall": agg(records, recipe, "jllm_recall"),
                "jllm_nyes": agg(records, recipe, "jllm_nyes"),
                "menu_hit": agg(records, recipe, "menu_hit")}

    direct, indirect = rp("direct"), rp("indirect")
    allrec = rp("direct")  # placeholder overwritten below
    for k in allrec:
        vals = [r[k] for r in records]
        allrec[k] = float(np.mean(vals)) if vals else None
    abst = {a: float(np.mean([n[a] for n in negs])) for a in
            ("V", "J-combo", "J-llm", "J-menu")}
    ece_combo, ece_jllm = ece(*zip(*ece_c)), ece(*zip(*ece_j)) if ece_j \
        else None

    gap_ind = indirect["jllm_recall"] - indirect["v_p3"]
    direct_ok = abs(direct["jllm_recall"] - direct["v_p3"]) <= 0.03
    guard = abst["J-llm"] >= 0.5
    if gap_ind >= 0.10 and direct_ok and guard:
        r1 = "PASS"
    elif gap_ind <= 0.03 or (direct["jllm_recall"] - direct["v_p3"]) < -0.03:
        r1 = "FAIL"
    else:
        r1 = "PARTIAL"
    denom = indirect["jllm_recall"] - indirect["v_p3"]
    keep = ((indirect["c_p3"] - indirect["v_p3"]) / denom
            if denom and denom > 0 else None)
    r2 = ("PASS" if keep is not None and keep >= 0.5 else
          "FAIL" if keep is not None else "n/a")
    r3 = ("PASS" if allrec["jllm_recall"] - allrec["menu_hit"] >= -0.02
          else "FAIL")
    r4 = ("PASS" if max(x for x in (ece_combo, ece_jllm) if x is not None)
          >= 0.05 else "FAIL") if (ece_combo or ece_jllm) else "n/a"

    out = {"tag": "chunkrel", "commit": git_commit(),
           "model_fp": EC.fingerprint(), "tau": tau, "tau_train_acc": tau_acc,
           "combo_val_ce": val_ce, "p_yes_coverage":
           round(p_cover[0] / max(p_cover[1], 1), 4),
           "by_recipe": {"direct": direct, "indirect": indirect,
                         "all": allrec},
           "abstain": abst, "negatives": negs, "records": records,
           "ece": {"J-combo": ece_combo, "J-llm": ece_jllm},
           "usage": {k: {"prompt_tokens": v[0],
                         "completion_tokens": v[1],
                         "ms_per_q": round(v[2] / max(v[3], 1), 1)}
                     for k, v in usage.items()},
           "verdicts": {"R1": r1, "R2": r2, "R3": r3, "R4": r4,
                        "gap_indirect": round(gap_ind, 4),
                        "combo_keep_fraction": (round(keep, 4)
                                                if keep is not None
                                                else None)}}
    with open(os.path.join(RES, "chunkrel.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {os.path.join(RES, 'chunkrel.json')}")
    print(json.dumps({"by_recipe": out["by_recipe"], "abstain": abst,
                      "verdicts": out["verdicts"], "ece": out["ece"]},
                     indent=1))


# ----------------------------------------------------------------- smoke

SMOKE_QS = [
    {"recipe": "direct",
     "q": "What is WireScale and what rule does its laboratory follow?",
     "chunk": 0},
    {"recipe": "indirect",
     "q": "Where do I look to decide between asking small questions step "
          "by step and asking one big question over everything?",
     "chunk": 1},
]


def run_smoke():
    chunkmap = {"README.md": build_chunkmap()["README.md"][:5]}
    ch = chunkmap["README.md"]
    cache = EC.load_cache()
    texts = ([e["q"] for e in SMOKE_QS] + [HAND_NEGS[0]] + ch)
    mats = embed_all(chunkmap, texts, cache)
    M = mats["README.md"]
    tiny_pairs = []
    for e in SMOKE_QS:
        tiny_pairs.append((e["q"], ch[e["chunk"]], 1, "README.md",
                           e["chunk"], e))
        for i in range(len(ch)):
            if i != e["chunk"]:
                tiny_pairs.append((e["q"], ch[i], 0, "README.md", i, e))
    net, vce = train_combo(tiny_pairs, cache, epochs=8)
    out = {"smoke": True, "commit": git_commit(),
           "model_fp": EC.fingerprint(), "n_chunks": len(ch),
           "combo_val_ce": vce, "questions": []}
    for e in SMOKE_QS:
        qv = vec(cache, e["q"])
        order = np.argsort(-(M @ qv))
        with torch.no_grad():
            F = np.stack([feats(qv, M[i]) for i in range(len(ch))])
            p = torch.sigmoid(net(torch.from_numpy(F))).numpy()
        corder = np.argsort(-p)
        yes_set = [i for i, c in enumerate(ch) if judge(e["q"], c)[0]]
        idxs, prev = menu_subset("README.md", e["chunk"], chunkmap)
        k, _ = menu_pick(e["q"], prev)
        out["questions"].append({
            "q": e["q"], "gold": e["chunk"], "v_rank":
            int(np.nonzero(order == e["chunk"])[0][0]),
            "c_rank": int(np.nonzero(corder == e["chunk"])[0][0]),
            "jllm_yes_set": yes_set, "menu_pick": idxs[k - 1] if k else -1})
        log(f"  smoke {e['recipe']}: v_rank={out['questions'][-1]['v_rank']} "
            f"c_rank={out['questions'][-1]['c_rank']} "
            f"jllm={yes_set} menu={out['questions'][-1]['menu_pick']}")
    qn = HAND_NEGS[0]
    qv = vec(cache, qn)
    yes_neg = [i for i, c in enumerate(ch) if judge(qn, c)[0]]
    with torch.no_grad():
        F = np.stack([feats(qv, M[i]) for i in range(len(ch))])
        pn = torch.sigmoid(net(torch.from_numpy(F))).numpy()
    idxs, prev = menu_subset("README.md", 0, chunkmap)
    out["negative"] = {"q": qn, "v_top_cos": round(float((M @ qv).max()), 4),
                       "jllm_yes_set": yes_neg,
                       "combo_p_max": round(float(pn.max()), 4),
                       "menu_pick": menu_pick(qn, prev)[0]}
    with open(os.path.join(RES, "chunkrel_smoke.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log(f"wrote {os.path.join(RES, 'chunkrel_smoke.json')} (SMOKE — "
        f"never scored)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gen-questions", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
    elif args.gen_questions:
        chunkmap = build_chunkmap()
        log(f"docs: { {d: len(c) for d, c in chunkmap.items()} }")
        qdata = gen_questions(chunkmap)
        with open(QFILE, "w") as f:
            json.dump(qdata, f, indent=1, sort_keys=True)
        log(f"wrote {QFILE} — HAND-AUDIT before any scored run "
            f"(protocol R-2); freeze by committing.")
    else:
        run_scored()
    return 0


if __name__ == "__main__":
    sys.exit(main())
