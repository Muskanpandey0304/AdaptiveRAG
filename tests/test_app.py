"""Smoke tests for app.py using Streamlit's headless AppTest harness (no
real browser needed).

The out-of-scope test injects a small synthetic corpus in place of the
real Retriever, same as tests/test_pipeline.py, since data/raw/ has no
real PDFs yet.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import src.pipeline as pipeline
from src.indexer import build_vector_index, compute_centroid
from src.llm import LLMResult
from src.retrievers import Retriever

pytestmark = pytest.mark.slow  # builds a real embedding index (downloads the model)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_direct_route_without_a_retriever(monkeypatch) -> None:
    """A greeting routes "direct" and never touches the retriever, so the
    app should render end to end with just the LLM mocked."""

    def fake_generate(prompt: str, system: str | None = None) -> LLMResult:
        return LLMResult(text="Hello!", input_tokens=1, output_tokens=1, latency_s=0.01)

    monkeypatch.setattr("src.generator.generate", fake_generate)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].input("Hi there!")
    at.button[0].click().run()

    assert not at.exception
    assert any("Hello!" in md.value for md in at.markdown)
    assert any("direct" in md.value for md in at.markdown)


def _fastapi_chunks() -> list[dict]:
    topics = [
        "The FastAPI framework uses Python type hints for request validation.",
        "Uvicorn is an ASGI server commonly used to run FastAPI applications.",
        "Pydantic models define the schema for FastAPI request and response bodies.",
    ]
    return [{"id": i, "text": text, "source": "fastapi.pdf", "page": 1} for i, text in enumerate(topics)]


@pytest.fixture
def synthetic_retriever(tmp_path, monkeypatch) -> Retriever:
    chunks = _fastapi_chunks()
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


def test_app_shows_not_found_for_out_of_scope_question(synthetic_retriever) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()
    # Reranking is on by default in the app; the first run loads the ~1GB
    # cross-encoder, which takes far longer than AppTest's 3s default.
    at.text_input[0].input("What is the boiling point of mercury on Mars?")
    at.button[0].click().run(timeout=180)

    assert not at.exception
    assert any("Not found in the documents." in md.value for md in at.markdown)


def test_app_does_not_call_the_llm_again_when_a_widget_changes(monkeypatch) -> None:
    """Regression: the old app reran the whole pipeline on every widget change."""
    calls = []

    def fake_generate(prompt: str, system: str | None = None) -> LLMResult:
        calls.append(1)
        return LLMResult(text="Hello!", input_tokens=1, output_tokens=1, latency_s=0.01)

    monkeypatch.setattr("src.generator.generate", fake_generate)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].input("Hi there!")
    at.button[0].click().run()
    assert len(calls) == 1

    at.checkbox[0].uncheck().run()  # a widget change: must not re-ask
    assert len(calls) == 1
    assert any("Hello!" in md.value for md in at.markdown)  # answer still shown


def test_app_shows_a_friendly_error_when_ollama_is_down(monkeypatch) -> None:
    def down(*args, **kwargs):
        raise ConnectionError("no server")

    monkeypatch.setattr("src.generator.generate", down)
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].input("Hi there!")
    at.button[0].click().run()

    assert not at.exception
    assert any("Could not reach Ollama" in e.value for e in at.error)
