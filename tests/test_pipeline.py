"""Tests for src/pipeline.py.

The out-of-domain test builds a small synthetic (FastAPI-flavored) index
and injects it in place of the real Retriever, since data/raw/ has no real
PDFs yet. Asking something far outside that corpus should drive confidence
below VECTOR_MIN_SIM, so the LLM is never called and "Not found" comes
back directly -- not an invented answer.
"""

import json

import faiss
import numpy as np
import pytest

import src.pipeline as pipeline
from src.indexer import build_vector_index, compute_centroid
from src.llm import LLMResult
from src.retrievers import Retriever


def _fastapi_chunks() -> list[dict]:
    topics = [
        "The FastAPI framework uses Python type hints for request validation.",
        "Uvicorn is an ASGI server commonly used to run FastAPI applications.",
        "Pydantic models define the schema for FastAPI request and response bodies.",
        "Dependency injection in FastAPI is done with the Depends function.",
        "SQLAlchemy is a popular ORM often paired with FastAPI for databases.",
    ]
    return [{"id": i, "text": text, "source": "fastapi.pdf", "page": 1} for i, text in enumerate(topics)]


def _build_retriever_with_centroid(tmp_path, chunks: list[dict], monkeypatch) -> Retriever:
    chunks_path = tmp_path / "chunks.json"
    faiss_path = tmp_path / "faiss.index"
    centroid_path = tmp_path / "centroid.npy"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    vector_index = build_vector_index(chunks)
    faiss.write_index(vector_index, str(faiss_path))
    np.save(centroid_path, compute_centroid(vector_index))

    retriever = Retriever(
        chunks_path=chunks_path, faiss_path=faiss_path, centroid_path=centroid_path
    )
    monkeypatch.setattr(pipeline, "_retriever", retriever)
    return retriever


@pytest.fixture
def synthetic_retriever(tmp_path, monkeypatch) -> Retriever:
    return _build_retriever_with_centroid(tmp_path, _fastapi_chunks(), monkeypatch)


def test_out_of_domain_question_returns_not_found(synthetic_retriever) -> None:
    result = pipeline.ask("What is the boiling point of mercury on Mars?", force_route="hybrid")
    assert result.answer.text == pipeline.NOT_FOUND_TEXT
    assert result.answer.citations == []
    # Confidence-gated miss: no LLM call should have happened.
    assert result.answer.tokens_in == 0
    assert result.answer.tokens_out == 0


def _wide_fastapi_chunks() -> list[dict]:
    """A bigger same-source corpus than TOP_K, so there's room for
    expand_neighbors to actually add ids beyond the original top-TOP_K."""
    topics = [
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        "It is based on standard Python type hints for declaring request and response data.",
        "FastAPI uses Pydantic models to validate incoming request bodies automatically.",
        "Uvicorn is the ASGI server most commonly used to run a FastAPI application.",
        "Starlette provides the underlying ASGI toolkit that FastAPI is built on top of.",
        "Dependency injection in FastAPI is handled through the Depends function.",
        "Path parameters and query parameters are declared directly as function arguments.",
        "FastAPI automatically generates interactive OpenAPI and Swagger documentation.",
        "Background tasks let a FastAPI endpoint return a response before finishing work.",
        "Middleware in FastAPI can process every request before it reaches a route handler.",
        "SQLAlchemy is a common ORM choice for FastAPI applications that need a database.",
        "Testing a FastAPI app is usually done with Starlette's TestClient.",
    ]
    return [{"id": i, "text": text, "source": "fastapi.pdf", "page": 1} for i, text in enumerate(topics)]


@pytest.fixture
def wide_synthetic_retriever(tmp_path, monkeypatch) -> Retriever:
    return _build_retriever_with_centroid(tmp_path, _wide_fastapi_chunks(), monkeypatch)


def test_expand_neighbors_adds_hits_beyond_top_k(wide_synthetic_retriever, monkeypatch) -> None:
    def fake_generate(prompt: str, system: str | None = None) -> LLMResult:
        return LLMResult(text="Some answer [1].", input_tokens=1, output_tokens=1, latency_s=0.01)

    monkeypatch.setattr("src.generator.generate", fake_generate)

    without = pipeline.ask("What ASGI server runs FastAPI?", force_route="hybrid", expand_neighbors=False)
    with_expansion = pipeline.ask("What ASGI server runs FastAPI?", force_route="hybrid", expand_neighbors=True)

    assert len(with_expansion.hits) > len(without.hits)
    assert any(hit.meta.get("neighbor_of") is not None for hit in with_expansion.hits)
    # Neighbor expansion must never change the confidence that was already
    # checked against VECTOR_MIN_SIM before expansion happened.
    assert with_expansion.confidence == without.confidence


def _chunks_with_decoy() -> list[dict]:
    """A FastAPI corpus with one decoy chunk that happens to contain an
    off-topic benchmark-style Q&A pair -- the same shape as the real
    "What is the capital of France?" failure found during evaluation."""
    topics = [
        "FastAPI is a modern Python web framework for building APIs.",
        "Uvicorn is the ASGI server used to run FastAPI applications.",
        "Pydantic models validate FastAPI request and response bodies.",
        "Dependency injection in FastAPI is handled by the Depends function.",
        "FastAPI automatically generates OpenAPI and Swagger documentation.",
        "Example benchmark question: What is the capital of France? Answer: Paris.",
    ]
    return [{"id": i, "text": text, "source": "fastapi.pdf", "page": 1} for i, text in enumerate(topics)]


@pytest.fixture
def decoy_retriever(tmp_path, monkeypatch) -> Retriever:
    return _build_retriever_with_centroid(tmp_path, _chunks_with_decoy(), monkeypatch)


def test_centroid_gate_catches_a_decoy_chunk_despite_high_per_chunk_confidence(decoy_retriever) -> None:
    """The decoy chunk should score a high per-chunk match for this
    off-topic query (it nearly contains the question verbatim), but the
    corpus's overall topic is FastAPI, not geography -- the centroid gate
    should catch what the per-chunk confidence alone would miss."""
    result = pipeline.ask("What is the capital of France?", force_route="hybrid")
    assert result.answer.text == pipeline.NOT_FOUND_TEXT
    assert result.answer.tokens_in == 0  # no LLM call -- caught before generation


def test_force_route_direct_bypasses_router(monkeypatch) -> None:
    def fake_generate(prompt: str, system: str | None = None) -> LLMResult:
        return LLMResult(text="Hi!", input_tokens=1, output_tokens=1, latency_s=0.01)

    monkeypatch.setattr("src.generator.generate", fake_generate)

    # This query would normally route to "hybrid" (a long conceptual
    # question); force_route should bypass the router entirely.
    result = pipeline.ask(
        "Why would a chunk's confidence score be low even if it looks relevant?",
        force_route="direct",
    )
    assert result.route == "direct"
    assert result.reason == "forced"
    assert result.escalated is False


def test_rerank_sees_a_wider_pool_than_top_k(wide_synthetic_retriever, monkeypatch) -> None:
    """Regression: reranking used to receive only the TOP_K hybrid hits, so
    it could only reshuffle them. It must get the wider RERANK_POOL_K pool."""
    seen_pool_sizes = []
    real_rerank = wide_synthetic_retriever.rerank

    def spy(query, hits, k):
        seen_pool_sizes.append(len(hits))
        return real_rerank(query, hits, k)

    monkeypatch.setattr(wide_synthetic_retriever, "rerank", spy)
    monkeypatch.setattr("src.pipeline.RERANK_MIN_PROB", 0.0)
    monkeypatch.setattr(
        "src.generator.generate",
        lambda prompt, system=None: LLMResult(text="ok [1]", input_tokens=1, output_tokens=1, latency_s=0.0),
    )

    result = pipeline.ask("What ASGI server runs FastAPI?", force_route="hybrid", use_rerank=True)

    assert seen_pool_sizes and seen_pool_sizes[0] > pipeline.TOP_K  # 12 chunks in the fixture
    assert len(result.hits) == pipeline.TOP_K


def test_auto_context_mode_trims_lookups_but_not_explanations(synthetic_retriever, monkeypatch) -> None:
    """CONTEXT_MODE="auto": a single-fact lookup is sent through sentence
    selection; an explanatory question gets the full parents."""
    monkeypatch.setattr("src.config.CONTEXT_MODE", "auto")
    calls = []
    real = synthetic_retriever.select_sentences
    monkeypatch.setattr(synthetic_retriever, "select_sentences", lambda q, h: (calls.append(q), real(q, h))[1])
    monkeypatch.setattr(
        "src.generator.generate",
        lambda p, system=None, schema=None: LLMResult(text="ok [1]", input_tokens=1, output_tokens=1, latency_s=0.0),
    )
    hits = synthetic_retriever.vector_search("ASGI server FastAPI", 3)

    pipeline._generate_answer("How many ASGI servers does the text name?", hits)
    pipeline._generate_answer("Why is Uvicorn used to run FastAPI?", hits)

    assert calls == ["How many ASGI servers does the text name?"]
