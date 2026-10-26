"""Tests for src/retrievers.py.

Built from small synthetic chunks (real chunks.json/faiss.index/bm25.pkl
don't exist yet -- data/raw/ is still empty) so the retrieval *algorithms*
(sorting, k, dedup, RRF's rank-overlap property, reranking) can be checked
without real project documents.
"""

import json
import pickle

import faiss
import numpy as np
import pytest

from src.indexer import build_bm25_index, build_vector_index, compute_centroid
from src.retrievers import Hit, Retriever, _fuse_rrf


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


def _build_retriever(tmp_path, chunks: list[dict], with_centroid: bool = True) -> Retriever:
    chunks_path = tmp_path / "chunks.json"
    faiss_path = tmp_path / "faiss.index"
    bm25_path = tmp_path / "bm25.pkl"
    centroid_path = tmp_path / "centroid.npy"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    vector_index = build_vector_index(chunks)
    faiss.write_index(vector_index, str(faiss_path))
    with open(bm25_path, "wb") as f:
        pickle.dump(build_bm25_index(chunks), f)
    if with_centroid:
        np.save(centroid_path, compute_centroid(vector_index))

    return Retriever(chunks_path=chunks_path, faiss_path=faiss_path, bm25_path=bm25_path, centroid_path=centroid_path)


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


def test_expand_neighbors_adds_adjacent_same_source_chunks(retriever) -> None:
    anchor = Hit(chunk_id=3, score=0.9, chunk=retriever.chunks[3])
    expanded = retriever.expand_neighbors([anchor], window=1)
    ids = [h.chunk_id for h in expanded]
    assert ids == [2, 3, 4]
    assert expanded[0].meta.get("neighbor_of") == 3
    assert expanded[1] is anchor
    assert expanded[2].meta.get("neighbor_of") == 3


def test_expand_neighbors_dedupes_overlapping_clusters(retriever) -> None:
    # chunk 3 is already in chunk 2's neighbor window, so it must appear
    # exactly once in the output, not once as a neighbor and once as its
    # own anchor.
    hit_a = Hit(chunk_id=2, score=0.9, chunk=retriever.chunks[2])
    hit_b = Hit(chunk_id=3, score=0.8, chunk=retriever.chunks[3])
    expanded = retriever.expand_neighbors([hit_a, hit_b], window=1)
    ids = [h.chunk_id for h in expanded]
    assert len(ids) == len(set(ids))
    assert {2, 3}.issubset(set(ids))


def test_expand_neighbors_does_not_cross_source_boundary(tmp_path) -> None:
    chunks = [
        {"id": 0, "text": "last chunk of document A talks about endings.", "source": "a.pdf", "page": 5},
        {"id": 1, "text": "first chunk of document B talks about beginnings.", "source": "b.pdf", "page": 1},
        {"id": 2, "text": "second chunk of document B talks about middles.", "source": "b.pdf", "page": 1},
    ]
    boundary_retriever = _build_retriever(tmp_path, chunks)
    anchor = Hit(chunk_id=1, score=0.5, chunk=chunks[1])

    expanded = boundary_retriever.expand_neighbors([anchor], window=1)

    ids = [h.chunk_id for h in expanded]
    assert 0 not in ids  # adjacent id, but a different source -- must not be pulled in
    assert ids == [1, 2]


def test_centroid_similarity_higher_for_on_topic_query(retriever) -> None:
    on_topic = retriever.centroid_similarity("What ASGI server does FastAPI use?")
    off_topic = retriever.centroid_similarity("What is the boiling point of mercury on Mars?")
    assert on_topic > off_topic


def test_centroid_similarity_raises_without_a_built_centroid(tmp_path) -> None:
    retriever_without_centroid = _build_retriever(tmp_path, _chunks(), with_centroid=False)
    with pytest.raises(RuntimeError):
        retriever_without_centroid.centroid_similarity("anything")


def _parent_child_retriever(tmp_path) -> Retriever:
    """Two parents (ids 0, 1), each split into two children (child ids 0-3)."""
    parents = [
        {"id": 0, "text": "Alpha one. Alpha two. Beta one. Beta two.", "source": "a.pdf", "page": 1},
        {"id": 1, "text": "Gamma one. Gamma two. Delta one. Delta two.", "source": "a.pdf", "page": 2},
    ]
    children = [
        {"id": 0, "text": "Alpha one. Alpha two.", "source": "a.pdf", "page": 1, "parent_id": 0},
        {"id": 1, "text": "Beta one. Beta two.", "source": "a.pdf", "page": 1, "parent_id": 0},
        {"id": 2, "text": "Gamma one. Gamma two.", "source": "a.pdf", "page": 2, "parent_id": 1},
        {"id": 3, "text": "Delta one. Delta two.", "source": "a.pdf", "page": 2, "parent_id": 1},
    ]
    retriever = _build_retriever(tmp_path, children)
    with open(tmp_path / "parents.json", "w", encoding="utf-8") as f:
        json.dump(parents, f)
    return Retriever(
        chunks_path=tmp_path / "chunks.json",
        faiss_path=tmp_path / "faiss.index",
        bm25_path=tmp_path / "bm25.pkl",
        centroid_path=tmp_path / "centroid.npy",
    )


def test_to_parents_collapses_siblings_and_returns_parent_text(tmp_path) -> None:
    retriever = _parent_child_retriever(tmp_path)
    children = [
        Hit(chunk_id=1, score=0.9, chunk=retriever.chunks[1], meta={"vector_sim": 0.4}),
        Hit(chunk_id=0, score=0.8, chunk=retriever.chunks[0], meta={"vector_sim": 0.7}),
        Hit(chunk_id=3, score=0.7, chunk=retriever.chunks[3], meta={"vector_sim": 0.5}),
    ]
    parents = retriever.to_parents(children)

    assert [h.chunk_id for h in parents] == [0, 1]  # siblings 1 and 0 -> one parent, best-ranked first
    assert parents[0].chunk["text"] == retriever.parents[0]["text"]  # parent text, not the child's
    assert parents[0].score == 0.9  # score of the best-ranked child
    assert parents[0].meta["child_id"] == 1
    assert parents[0].meta["vector_sim"] == 0.7  # best cosine among siblings


def test_to_parents_is_a_noop_without_parents(retriever) -> None:
    hits = retriever.vector_search("FastAPI", k=3)
    assert retriever.to_parents(hits) == hits


def test_rerank_scores_are_probabilities(retriever) -> None:
    candidates = retriever.hybrid_search("FastAPI dependency injection", k=5)
    reranked = retriever.rerank("FastAPI dependency injection", candidates, k=5)
    assert all(0.0 <= h.score <= 1.0 for h in reranked)


def test_select_sentences_keeps_the_relevant_sentence_and_its_neighbors(tmp_path) -> None:
    text = (
        "The weather was mild that year. Farmers planted early crops. "
        "The ASGI server Uvicorn is used to run FastAPI applications. "
        "Bread prices rose slowly. Tea was popular in the north."
    )
    chunks = [{"id": 0, "text": text, "source": "a.pdf", "page": 3}]
    r = _build_retriever(tmp_path, chunks)
    hit = Hit(chunk_id=0, score=0.9, chunk=chunks[0], meta={"vector_sim": 0.8})

    out = r.select_sentences("What ASGI server runs FastAPI?", [hit], top_n=1, window=1)

    assert len(out) == 1
    assert "Uvicorn" in out[0].chunk["text"]
    assert "Tea was popular" not in out[0].chunk["text"]  # outside the window
    assert len(out[0].chunk["text"]) < len(text)
    assert (out[0].chunk["source"], out[0].chunk["page"], out[0].chunk_id) == ("a.pdf", 3, 0)
    assert out[0].meta["compressed"] is True and out[0].meta["vector_sim"] == 0.8
    assert hit.chunk["text"] == text  # the input hit is not modified
