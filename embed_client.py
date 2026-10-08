"""Batched embeddings on 127.0.0.1:8998, id_slot 7 ONLY (production readout
is slot 6 — never touched). Disk cache keyed sha1(text)+model fingerprint;
fp16; cache is gitignored and must be dropped if the serving model changes."""
import hashlib
import json
import os
import pickle
import urllib.request

BASE = "http://127.0.0.1:8998"
SLOT = 7
BATCH = 32
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "cache")


def _post(path, payload, timeout=120):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fingerprint():
    with urllib.request.urlopen(BASE + "/props", timeout=10) as r:
        p = json.loads(r.read())
    return hashlib.sha1(p.get("model_path", "unknown").encode()).hexdigest()[:12]


def _cache_path():
    return os.path.join(CACHE_DIR, f"emb_{fingerprint()}.pkl")


def load_cache():
    try:
        with open(_cache_path(), "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def embed(texts, cache, save_every=2000):
    """Return dict text -> np.float16[2560]; fills `cache` in place."""
    import numpy as np
    todo = sorted({t for t in texts if t not in cache})
    new = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        resp = _post("/v1/embeddings", {"input": chunk, "model": "tree-lab",
                                        "id_slot": SLOT})
        for item, text in zip(resp["data"], chunk):
            cache[text] = np.asarray(item["embedding"], dtype=np.float16)
        new += len(chunk)
        if new and new % (save_every * 10) == 0:
            save_cache(cache)
    if new:
        save_cache(cache)
    return cache


def save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _cache_path() + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    os.replace(tmp, _cache_path())


if __name__ == "__main__":
    c = load_cache()
    before = len(c)
    probe = ["turn the kitchen lamps to 40 percent"]
    embed(probe, c)
    v = c[probe[0]]
    n = float(np := __import__("numpy").linalg.norm(v.astype("float32")))
    print(f"cache {before} -> {len(c)}; dim={v.shape[0]} L2={n:.3f} "
          f"fp={fingerprint()}")
