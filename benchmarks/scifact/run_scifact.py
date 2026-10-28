"""Run this project's retrieval stack on BEIR SciFact, an outside benchmark.

SciFact: 5,183 scientific abstracts and 300 test claims; a claim's relevant
abstracts are labelled by experts. It tests retrieval only (no LLM), scored with
nDCG@10 like the published BEIR baselines. Nothing here touches the project's
own index: the benchmark has its own folder, corpus, embeddings and results.

The same components as the main pipeline are used, with the same settings:
e5-base-v2 embeddings (query/passage prefixes), BM25 over the same tokenizer,
RRF fusion of the top CANDIDATES_K from each, and a bge-reranker-base rerank of
the top RERANK_POOL_K fused candidates. The abstracts are short (about one
chunk), so they are indexed whole instead of being chunked, and the title is
prepended exactly as the pipeline does for papers.

    PYTHONPATH=. python benchmarks/scifact/run_scifact.py
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import CANDIDATES_K, DOC_PREFIX, EMBED_MODEL, QUERY_PREFIX, RERANK_MODEL, RERANK_POOL_K
from src.eval import bootstrap_ci
from src.indexer import tokenize
from src.retrievers import _fuse_rrf

HERE = Path(__file__).parent
DATA = HERE / "data" / "scifact"
INDEX = HERE / "index"
K = 10


def load() -> tuple[list[str], list[str], dict[str, str], dict[str, set[str]]]:
    """(doc ids, doc texts, query id -> text, query id -> relevant doc ids) for the test split."""
    ids, texts = [], []
    with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["_id"])
            texts.append(f"{doc['title']}\n{doc['text']}")
    qrels: dict[str, set[str]] = {}
    with open(DATA / "qrels" / "test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    queries = {}
    with open(DATA / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]
    return ids, texts, queries, qrels


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    """Binary-relevance nDCG@k (log2 discount), as in the BEIR evaluation."""
    dcg = sum(1.0 / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    """Fraction of the relevant documents found in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def embed_corpus(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Passage embeddings, cached on disk (about 10 minutes on CPU the first time)."""
    path = INDEX / f"doc_embeddings_{EMBED_MODEL.replace('/', '_')}.npy"
    if path.exists():
        return np.load(path)
    INDEX.mkdir(parents=True, exist_ok=True)
    start = time.time()
    vectors = model.encode(
        [DOC_PREFIX + t for t in texts], batch_size=16, normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")
    np.save(path, vectors)
    print(f"embedded {len(texts)} docs in {time.time() - start:.0f}s", flush=True)
    return vectors


def main() -> None:
    ids, texts, queries, qrels = load()
    print(f"{len(ids)} docs, {len(queries)} test queries", flush=True)

    model = SentenceTransformer(EMBED_MODEL)
    doc_vectors = embed_corpus(model, texts)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    reranker = CrossEncoder(RERANK_MODEL)

    per_query: dict[str, dict[str, list[float]]] = {
        name: {"ndcg": [], "recall": []} for name in ("BM25", "Dense (e5-base-v2)", "Hybrid (RRF)", "Hybrid + rerank")
    }
    start = time.time()
    for n, (qid, query) in enumerate(queries.items(), 1):
        relevant = qrels[qid]
        qvec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        dense_scores = doc_vectors @ qvec
        bm25_scores = bm25.get_scores(tokenize(query))

        dense_rank = [int(i) for i in np.argsort(dense_scores)[::-1][:CANDIDATES_K]]
        bm25_rank = [int(i) for i in np.argsort(bm25_scores)[::-1][:CANDIDATES_K]]
        fused = _fuse_rrf([dense_rank, bm25_rank])
        hybrid_rank = sorted(fused, key=lambda i: fused[i], reverse=True)[:RERANK_POOL_K]
        rerank_scores = reranker.predict([(query, texts[i]) for i in hybrid_rank])
        reranked = [hybrid_rank[i] for i in np.argsort(rerank_scores)[::-1]]

        rankings = {"BM25": bm25_rank, "Dense (e5-base-v2)": dense_rank, "Hybrid (RRF)": hybrid_rank, "Hybrid + rerank": reranked}
        for name, order in rankings.items():
            docs = [ids[i] for i in order]
            per_query[name]["ndcg"].append(ndcg_at_k(docs, relevant))
            per_query[name]["recall"].append(recall_at_k(docs, relevant))
        if n % 50 == 0:
            print(f"{n}/{len(queries)} queries, {time.time() - start:.0f}s", flush=True)

    summary = {}
    lines = [
        "# SciFact (BEIR) results",
        "",
        f"{len(ids)} abstracts, {len(queries)} test claims. Same retrieval components and settings as the main",
        f"pipeline ({EMBED_MODEL}, BM25, RRF, {RERANK_MODEL}); abstracts indexed whole. nDCG@10 / Recall@10 with",
        "95% bootstrap intervals over queries.",
        "",
        "| Method | nDCG@10 | 95% CI | Recall@10 |",
        "|---|---|---|---|",
    ]
    for name, m in per_query.items():
        mean, low, high = bootstrap_ci(m["ndcg"])
        recall = float(np.mean(m["recall"]))
        summary[name] = {"ndcg@10": mean, "ci": [low, high], "recall@10": recall}
        lines.append(f"| {name} | {mean:.3f} | [{low:.3f}, {high:.3f}] | {recall:.3f} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
