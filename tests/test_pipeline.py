"""Tests for src/pipeline.py.

The out-of-domain test builds a small synthetic (FastAPI-flavored) index
and injects it in place of the real Retriever, since data/raw/ has no real
PDFs yet. Asking something far outside that corpus should drive confidence
below VECTOR_MIN_SIM, so the LLM is never called and "Not found" comes
back directly -- not an invented answer.
"""

import json
import pickle

import faiss
import pytest

import src.pipeline as pipeline
from src.indexer import build_bm25_index, build_vector_index
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


@pytest.fixture
def synthetic_retriever(tmp_path, monkeypatch) -> Retriever:
    chunks = _fastapi_chunks()
    chunks_path = tmp_path / "chunks.json"
    faiss_path = tmp_path / "faiss.index"
    bm25_path = tmp_path / "bm25.pkl"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    faiss.write_index(build_vector_index(chunks), str(faiss_path))
    with open(bm25_path, "wb") as f:
        pickle.dump(build_bm25_index(chunks), f)

    retriever = Retriever(chunks_path=chunks_path, faiss_path=faiss_path, bm25_path=bm25_path)
    monkeypatch.setattr(pipeline, "_retriever", retriever)
    return retriever


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
    chunks = _wide_fastapi_chunks()
    chunks_path = tmp_path / "chunks.json"
    faiss_path = tmp_path / "faiss.index"
    bm25_path = tmp_path / "bm25.pkl"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    faiss.write_index(build_vector_index(chunks), str(faiss_path))
    with open(bm25_path, "wb") as f:
        pickle.dump(build_bm25_index(chunks), f)

    retriever = Retriever(chunks_path=chunks_path, faiss_path=faiss_path, bm25_path=bm25_path)
    monkeypatch.setattr(pipeline, "_retriever", retriever)
    return retriever


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
