
import argparse, hashlib, json, os, platform, random, time
from datetime import datetime, timezone
import numpy as np
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

SEED = 42
EMB_MODEL = "BAAI/bge-small-en-v1.5"                  # retriever (identical in all conditions)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # relevance scorer used by the filter
GEN_MODEL = "llama3.1:8b"                              # generator served by Ollama (identical in all conditions)
OLLAMA_URL = "http://localhost:11434/api/chat"
NUM_CTX = 8192        # IMPORTANT: Ollama's default window is small; 10 chunks x 200 words would be silently truncated
NUM_PREDICT = 512     # same answer-length cap in all conditions
TOP_K = 10
CHUNK_WORDS, OVERLAP = 200, 40
DEDUP_COS = 0.95                                       # redundancy removal inside the filter
CONTROL_M = 4                                          # frozen in Appendix A; never derive from the test set

PROMPT = ("Answer the question using ONLY the context below. If the context is not sufficient, say so.\n\n"
          "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def log_environment():
    import sentence_transformers, datasets, torch
    return {"python": platform.python_version(), "sentence_transformers": sentence_transformers.__version__,
            "datasets": datasets.__version__, "torch": torch.__version__, "numpy": np.__version__,
            "emb_model": EMB_MODEL, "rerank_model": RERANK_MODEL, "gen_model": GEN_MODEL,
            "top_k": TOP_K, "chunk_words": CHUNK_WORDS, "overlap": OVERLAP,
            "dedup_cos": DEDUP_COS, "seed": SEED}


def chunk(text):
    w = text.split()
    step = CHUNK_WORDS - OVERLAP
    return [" ".join(w[i:i + CHUNK_WORDS]) for i in range(0, max(len(w), 1), step) if w[i:i + CHUNK_WORDS]]


def build_corpus(ds):
    """Pool every document of the split into one corpus (deduplicated), then chunk it."""
    seen, chunks = set(), []
    for row in ds:
        for doc in row["documents"]:
            h = hashlib.md5(doc.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            chunks.extend(chunk(doc))
    return chunks


def retrieve(q, emb, chunk_vecs, chunks):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    sims = chunk_vecs @ qv
    idx = np.argsort(-sims)[:TOP_K]
    return idx, sims[idx]


def noise_filter(q, idx, chunk_vecs, chunks, reranker, tau):
    """Keep chunks whose cross-encoder score >= tau, drop near-duplicates, always keep >=1 chunk."""
    scores = reranker.predict([(q, chunks[i]) for i in idx])
    order = np.argsort(-scores)
    kept = []
    for o in order:
        i = idx[o]
        if scores[o] < tau and kept:
            continue
        if any(float(chunk_vecs[i] @ chunk_vecs[j]) > DEDUP_COS for j in kept):
            continue
        kept.append(i)
    return kept, {int(idx[o]): float(scores[o]) for o in range(len(idx))}


def generate(q, ctx_chunks):
    body = {"model": GEN_MODEL, "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            "messages": [{"role": "user", "content": PROMPT.format(ctx="\n\n".join(ctx_chunks), q=q)}]}
    r = requests.post(OLLAMA_URL, json=body, timeout=300)
    r.raise_for_status()
    j = r.json()
    return j["message"]["content"], {"prompt_tokens": j.get("prompt_eval_count"), "gen_tokens": j.get("eval_count")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="techqa")      # check available configs on the dataset card
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tau", type=float, default=-2.0)  # frozen value (pilot on DEV split); never tune on test
    ap.add_argument("--control-m", type=int, default=CONTROL_M,
                    help="Frozen length-matched control size. Appendix A fixes this at 4.")
    ap.add_argument("--out", default="runs/out.jsonl")
    a = ap.parse_args()
    if a.control_m < 1:
        ap.error("--control-m must be at least 1")
    random.seed(SEED); np.random.seed(SEED)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    ds = load_dataset("rungalileo/ragbench", a.subset, split=a.split)
    print("columns:", ds.column_names)                  # verify field names (question, documents, response, ...)
    emb = SentenceTransformer(EMB_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    cache_dir = os.path.dirname(a.out) or "."
    vecs_cache = os.path.join(cache_dir, f"cache_{a.subset}_{a.split}_vecs.npy")
    chunks_cache = os.path.join(cache_dir, f"cache_{a.subset}_{a.split}_chunks.json")
    if os.path.exists(vecs_cache) and os.path.exists(chunks_cache):
        print(f"Loading cached embeddings from {vecs_cache}...")
        vecs = np.load(vecs_cache)
        with open(chunks_cache, "r", encoding="utf-8") as cf:
            chunks = json.load(cf)
        print(f"Loaded {len(chunks)} chunks and {vecs.shape} vectors from cache.")
    else:
        chunks = build_corpus(ds)
        print("corpus chunks:", len(chunks))
        vecs = emb.encode(chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        try:
            np.save(vecs_cache, vecs)
            with open(chunks_cache, "w", encoding="utf-8") as cf:
                json.dump(chunks, cf, ensure_ascii=False)
            print(f"Saved cache to {vecs_cache} and {chunks_cache}")
        except Exception as e:
            print(f"Cache save warning: {e}")

    rows = random.Random(SEED).sample(range(len(ds)), min(a.n, len(ds)))
    pre = []
    for r in rows:
        ex = ds[r]
        idx, dense_scores = retrieve(ex["question"], emb, vecs, chunks)
        kept, sc = noise_filter(ex["question"], idx, vecs, chunks, reranker, a.tau)
        dense_score_map = {int(i): float(s) for i, s in zip(idx, dense_scores)}
        pre.append((r, ex, idx, kept, sc, dense_score_map))
    observed_mean_kept = float(np.mean([len(p[3]) for p in pre]))
    m = a.control_m
    print(f"mean kept after filter (descriptive only): {observed_mean_kept:.2f}; frozen control top-m = {m}")

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if "condition" in d:
                    done.add((str(d["qid"]), d["condition"]))
        print(f"resuming: {len(done)} (qid, condition) pairs already in {a.out}")
    qids = sorted(str(p[1].get("id", p[0])) for p in pre)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "rungalileo/ragbench", "subset": a.subset,
                "dataset_fingerprint": getattr(ds, "_fingerprint", None),
                "split": a.split, "split_size": len(ds), "n": len(pre),
                "tau": a.tau, "dedup_cos": DEDUP_COS, "m": m,
                "control_m_source": "Appendix A frozen protocol", "observed_mean_filter_kept": observed_mean_kept,
                "num_ctx": NUM_CTX,
                "qid_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest()}
    print("manifest:", manifest)
    with open(a.out, "a", encoding="utf-8") as f:
        if not done:
            f.write(json.dumps({"env": log_environment(), **manifest}) + "\n")
        for r, ex, idx, kept, sc, dense_score_map in pre:
            qid = str(ex.get("id", r))
            conds = {"baseline": list(idx), "filtered": kept, "length_matched": list(idx[:m])}
            for name, ids in conds.items():
                if (qid, name) in done:
                    continue
                t0 = time.time()
                ctx = [chunks[i] for i in ids]
                ans, usage = generate(ex["question"], ctx)
                f.write(json.dumps({"qid": qid, "condition": name, "question": ex["question"],
                                    "reference": ex.get("response"), "contexts": ctx,
                                    "chunk_ids": [int(i) for i in ids], "answer": ans, **usage,
                                    "dense_scores": [dense_score_map[int(i)] for i in ids],
                                    "reranker_scores": [sc.get(int(i)) for i in ids],
                                    "latency_s": round(time.time() - t0, 2)}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
