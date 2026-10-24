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
