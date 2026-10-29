"""Phase 1 of the PeerQA benchmark: one index per paper, retrieve for every question inside its own paper.

PeerQA asks about a given paper, so each question searches only that paper's index (unlike the parked QASPER
run, which shared one index across papers). Same stack as the main pipeline: the project's chunker and
parent/child split, e5-base-v2, BM25, RRF fusion, and the title-aware bge reranker. Run with the GPU environment:

    EMBED_DEVICE=cuda PYTHONPATH=. venv_gpu/Scripts/python.exe benchmarks/peerqa/retrieve.py

Writes index/<paper>/ (gitignored), cache/hits.pkl (gitignored) and retrieval_results.json. Resumable:
paper indexes that already exist are reused.
"""

import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DOC_PREFIX, EMBED_DEVICE, EMBED_MODEL, RERANK_POOL_K, TOP_K
from src.indexer import chunks_fingerprint, compute_centroid
from src.ingest import chunk_pages, contextualize, load_markdown, make_children
from src.retrievers import Retriever

HERE = Path(__file__).parent
MD_DIR = HERE / "data" / "papers_md"
INDEX = HERE / "index"
CACHE = HERE / "cache"


def build_paper_index(pages: list[dict], model: SentenceTransformer, folder: Path) -> None:
    """Chunk one paper and write chunks/parents/faiss/centroid/meta under `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    parents = chunk_pages(pages)
    children = make_children(parents)
    vectors = model.encode(
        [DOC_PREFIX + contextualize(c) for c in children], normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype="float32"))
    (folder / "parents.json").write_text(json.dumps(parents, ensure_ascii=False), encoding="utf-8")
    (folder / "chunks.json").write_text(json.dumps(children, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(folder / "faiss.index"))
    np.save(folder / "centroid.npy", compute_centroid(index))
    meta = {"embed_model": EMBED_MODEL, "n_chunks": len(children), "chunks_sha256": chunks_fingerprint(folder / "chunks.json")}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def open_paper_retriever(paper_id: str) -> Retriever:
    folder = INDEX / paper_id
    return Retriever(
        chunks_path=folder / "chunks.json", faiss_path=folder / "faiss.index", centroid_path=folder / "centroid.npy"
    )


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_found(hits: list, evidence: list[str]) -> list[bool]:
    """For each gold evidence sentence: does any retrieved parent contain it (compared on its first 80 characters)?"""
    haystacks = [norm(h.chunk["text"]) for h in hits]
    return [any(norm(e)[:80] in text for text in haystacks) for e in evidence]


def main() -> None:
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    pages_by_paper: dict[str, list[dict]] = defaultdict(list)
    for page in load_markdown(MD_DIR):
        pages_by_paper[page["source"].removesuffix(".pdf")].append(page)

    embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    for paper_id, pages in pages_by_paper.items():
        if not (INDEX / paper_id / "meta.json").exists():
            build_paper_index(pages, embed_model, INDEX / paper_id)
    print(f"{len(pages_by_paper)} paper indexes ready", flush=True)

    reranker = None
    cached: dict[str, dict] = {}
    outcomes: dict[str, list[float]] = {"hybrid_any": [], "rerank_any": [], "rerank_recall": []}
    start = time.time()
    retrievers: dict[str, Retriever] = {}
    for n, q in enumerate(questions, 1):
        retriever = retrievers.get(q["paper_id"])
        if retriever is None:
            retriever = retrievers[q["paper_id"]] = open_paper_retriever(q["paper_id"])
            retriever._embed_model = embed_model
            if reranker is not None:
                retriever._reranker = reranker
        pool = retriever.hybrid_search(q["question"], RERANK_POOL_K)
        plain_parents = retriever.to_parents(pool)[:TOP_K]
        reranked = retriever.rerank(q["question"], pool, len(pool))
        reranker = retriever._reranker
        parents = retriever.to_parents(reranked)[:TOP_K]
        cached[q["question_id"]] = {"hits": parents, "top_rerank": reranked[0].score if reranked else 0.0}
        if q["answerable"] and q["evidence"]:
            plain = evidence_found(plain_parents, q["evidence"])
            found = evidence_found(parents, q["evidence"])
            outcomes["hybrid_any"].append(float(any(plain)))
            outcomes["rerank_any"].append(float(any(found)))
            outcomes["rerank_recall"].append(sum(found) / len(found))
        if n % 25 == 0:
            print(f"{n}/{len(questions)} questions, {time.time() - start:.0f}s", flush=True)

    CACHE.mkdir(exist_ok=True)
    with open(CACHE / "hits.pkl", "wb") as f:
        pickle.dump(cached, f)

    summary = {"n_questions": len(questions), "n_with_evidence": len(outcomes["rerank_any"]), "seconds": round(time.time() - start, 1)}
    for key, values in outcomes.items():  # intervals are computed by score.py (this GPU environment has no src.eval deps)
        summary[key] = {"mean": float(np.mean(values)), "per_question": values}
    (HERE / "retrieval_results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
