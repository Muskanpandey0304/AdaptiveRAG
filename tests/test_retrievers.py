"""Tests for src/retrievers.py.

Built from small synthetic chunks (real chunks.json/faiss.index/bm25.pkl
don't exist yet -- data/raw/ is still empty) so the retrieval *algorithms*
(sorting, k, dedup, RRF's rank-overlap property, reranking) can be checked
without real project documents.
"""

import json
import pickle

import faiss
import pytest

from src.indexer import build_bm25_index, build_vector_index
from src.retrievers import Retriever, _fuse_rrf


def _chunks() -> list[dict]:
    topics = [
        "The FastAPI framework uses Python type hints for request validation.",
        "Uvicorn is an ASGI server commonly used to run FastAPI applications.",
        "Pydantic models define the schema for FastAPI request and response bodies.",
        "Dependency injection in FastAPI is done with the Depends function.",
        "SQLAlchemy is a popular ORM often paired with FastAPI for databases.",
        "FastAPI automatically generates OpenAPI and Swagger documentation.",
        "Background tasks in FastAPI run after the response has been sent.",
        "Testing FastAPI apps is usually done with the TestClient from Starlette.",
    ]
    return [{"id": i, "text": text, "source": "fastapi.pdf", "page": 1} for i, text in enumerate(topics)]


def _build_retriever(tmp_path, chunks: list[dict]) -> Retriever:
    chunks_path = tmp_path / "chunks.json"
    faiss_path = tmp_path / "faiss.index"
    bm25_path = tmp_path / "bm25.pkl"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    faiss.write_index(build_vector_index(chunks), str(faiss_path))
    with open(bm25_path, "wb") as f:
        pickle.dump(build_bm25_index(chunks), f)

    return Retriever(chunks_path=chunks_path, faiss_path=faiss_path, bm25_path=bm25_path)


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("retriever_fixture")
    return _build_retriever(tmp_path, _chunks())


def test_vector_search_is_sorted_and_respects_k(retriever) -> None:
    hits = retriever.vector_search("What ASGI server runs FastAPI?", k=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_search_is_sorted_and_respects_k(retriever) -> None:
    hits = retriever.bm25_search("Depends dependency injection", k=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_has_no_duplicate_ids_and_respects_k(retriever) -> None:
    hits = retriever.hybrid_search("How does FastAPI validate requests?", k=4)
    assert len(hits) <= 4
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_hits_carry_vector_sim_in_meta(retriever) -> None:
    hits = retriever.hybrid_search("FastAPI documentation generation", k=3)
    for hit in hits:
        assert "vector_sim" in hit.meta


def test_rrf_favors_items_ranked_first_in_both_lists() -> None:
    scores = _fuse_rrf([[1, 2, 3], [1, 4, 5]])
    assert scores[1] > scores[2]
    assert scores[1] > scores[4]


def test_rerank_respects_k_and_sorts_best_first(retriever) -> None:
    candidates = retriever.hybrid_search("FastAPI dependency injection", k=5)
    reranked = retriever.rerank("FastAPI dependency injection", candidates, k=2)
    assert len(reranked) == 2
    scores = [h.score for h in reranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_handles_empty_hits(retriever) -> None:
    assert retriever.rerank("anything", [], k=3) == []
