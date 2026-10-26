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


def test_is_not_found_tolerates_citation_noise() -> None:
    from src.generator import is_not_found

    assert is_not_found("NOT_FOUND")
    assert is_not_found("  NOT_FOUND\n")
    assert is_not_found("[1] [2] [3] NOT_FOUND")  # observed small-model output
    assert not is_not_found("The answer is NOT_FOUND in table 3 [2].")
    assert not is_not_found("Paris")


def test_has_answer_text_rejects_bare_citations() -> None:
    from src.generator import has_answer_text

    assert not has_answer_text("[1]")
    assert not has_answer_text("[n] [3]")
    assert has_answer_text("[1] It is 37.7%.")


def test_v2_retries_once_on_a_bare_citation_and_sums_cost(monkeypatch) -> None:
    from src.llm import LLMResult

    monkeypatch.setattr("src.generator.PROMPT_VERSION", "v2")
    replies = iter(["[1]", "It is 37.7% [1]."])
    calls = []

    def fake(prompt, system=None):
        calls.append(prompt)
        return LLMResult(text=next(replies), input_tokens=10, output_tokens=2, latency_s=1.0)

    monkeypatch.setattr("src.generator.generate", fake)
    answer = answer_with_context("q?", [_hit(1, "a widget fact")])

    assert answer.text == "It is 37.7% [1]."
    assert len(calls) == 2 and "no answer text" in calls[1]
    assert (answer.tokens_in, answer.tokens_out, answer.latency_s) == (20, 4, 2.0)
    assert answer.citations and answer.citations[0]["n"] == 1


def test_v2_does_not_retry_a_real_answer_or_not_found(monkeypatch) -> None:
    from src.llm import LLMResult

    monkeypatch.setattr("src.generator.PROMPT_VERSION", "v2")
    for reply in ("Widgets are great [1].", "NOT_FOUND"):
        calls = []

        def fake(prompt, system=None, _r=reply):
            calls.append(1)
            return LLMResult(text=_r, input_tokens=1, output_tokens=1, latency_s=0.1)

        monkeypatch.setattr("src.generator.generate", fake)
        answer_with_context("q?", [_hit(1, "a widget fact")])
        assert len(calls) == 1


def test_v1_never_retries(monkeypatch) -> None:
    monkeypatch.setattr("src.generator.PROMPT_VERSION", "v1")
    calls = []
    real = _fake_generate("[1]")
    monkeypatch.setattr("src.generator.generate", lambda p, system=None: (calls.append(1), real(p, system))[1])
    answer_with_context("q?", [_hit(1, "a widget fact")])
    assert len(calls) == 1


def test_v1r_retries_a_bare_citation_but_keeps_the_original_prompt(monkeypatch) -> None:
    from src.generator import _CONTEXT_SYSTEM_PROMPT
    from src.llm import LLMResult

    monkeypatch.setattr("src.generator.PROMPT_VERSION", "v1r")
    replies = iter(["[2]", "It is 37.7% [1]."])
    systems = []

    def fake(prompt, system=None):
        systems.append(system)
        return LLMResult(text=next(replies), input_tokens=5, output_tokens=1, latency_s=0.5)

    monkeypatch.setattr("src.generator.generate", fake)
    answer = answer_with_context("q?", [_hit(1, "a widget fact")])

    assert answer.text == "It is 37.7% [1]."
    assert systems == [_CONTEXT_SYSTEM_PROMPT, _CONTEXT_SYSTEM_PROMPT]  # no reworded prompt
    assert (answer.tokens_in, answer.tokens_out, answer.latency_s) == (10, 2, 1.0)
