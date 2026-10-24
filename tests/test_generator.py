"""Tests for src/generator.py, using a mocked LLM (no live model needed).

Citation parsing and NOT_FOUND handling are pure text-processing logic, so
they're tested here against a fake generate() rather than a real LLM call.
"""

from src.generator import answer_direct, answer_with_context
from src.llm import LLMResult
from src.retrievers import Hit


def _hit(chunk_id: int, text: str, source: str = "doc.pdf", page: int = 1) -> Hit:
    return Hit(chunk_id=chunk_id, score=1.0, chunk={"id": chunk_id, "text": text, "source": source, "page": page})


def _fake_generate(text: str):
    """A stand-in for src.llm.generate that always returns `text`."""

    def _generate(prompt: str, system: str | None = None) -> LLMResult:
        return LLMResult(text=text, input_tokens=10, output_tokens=5, latency_s=0.01)

    return _generate


def test_answer_with_context_parses_valid_citations(monkeypatch) -> None:
    hits = [_hit(0, "Chunk A text", source="a.pdf", page=1), _hit(1, "Chunk B text", source="b.pdf", page=2)]
    monkeypatch.setattr("src.generator.generate", _fake_generate("Widgets are great [1]. Also see [2]."))

    answer = answer_with_context("What are widgets?", hits)

    assert answer.citations == [
        {"n": 1, "source": "a.pdf", "page": 1},
        {"n": 2, "source": "b.pdf", "page": 2},
    ]


def test_answer_with_context_ignores_out_of_range_citations(monkeypatch) -> None:
    hits = [_hit(0, "Chunk A text")]
    monkeypatch.setattr("src.generator.generate", _fake_generate("See [1] and also [5]."))

    answer = answer_with_context("question", hits)

    assert answer.citations == [{"n": 1, "source": "doc.pdf", "page": 1}]


def test_answer_with_context_deduplicates_repeated_citations(monkeypatch) -> None:
    hits = [_hit(0, "Chunk A text")]
    monkeypatch.setattr("src.generator.generate", _fake_generate("[1] and again [1]."))

    answer = answer_with_context("question", hits)

    assert answer.citations == [{"n": 1, "source": "doc.pdf", "page": 1}]


def test_answer_with_context_handles_not_found(monkeypatch) -> None:
    hits = [_hit(0, "Unrelated text")]
    monkeypatch.setattr("src.generator.generate", _fake_generate("NOT_FOUND"))

    answer = answer_with_context("unanswerable question", hits)

    assert answer.text == "NOT_FOUND"
    assert answer.citations == []


def test_answer_direct_has_no_citations(monkeypatch) -> None:
    monkeypatch.setattr("src.generator.generate", _fake_generate("Hi there!"))

    answer = answer_direct("hello")

    assert answer.citations == []
    assert answer.text == "Hi there!"
