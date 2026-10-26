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

    monkeypatch.setattr("src.config.PROMPT_VERSION", "v2")
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

    monkeypatch.setattr("src.config.PROMPT_VERSION", "v2")
    for reply in ("Widgets are great [1].", "NOT_FOUND"):
        calls = []

        def fake(prompt, system=None, _r=reply):
            calls.append(1)
            return LLMResult(text=_r, input_tokens=1, output_tokens=1, latency_s=0.1)

        monkeypatch.setattr("src.generator.generate", fake)
        answer_with_context("q?", [_hit(1, "a widget fact")])
        assert len(calls) == 1


def test_v1_never_retries(monkeypatch) -> None:
    monkeypatch.setattr("src.config.PROMPT_VERSION", "v1")
    calls = []
    real = _fake_generate("[1]")
    monkeypatch.setattr("src.generator.generate", lambda p, system=None: (calls.append(1), real(p, system))[1])
    answer_with_context("q?", [_hit(1, "a widget fact")])
    assert len(calls) == 1


def test_v1r_retries_a_bare_citation_but_keeps_the_original_prompt(monkeypatch) -> None:
    from src.generator import _CONTEXT_SYSTEM_PROMPT
    from src.llm import LLMResult

    monkeypatch.setattr("src.config.PROMPT_VERSION", "v1r")
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


def _json_reply(**fields) -> str:
    import json

    return json.dumps({"found": True, "answer": "", "sources": [], **fields})


def test_v3_turns_structured_json_into_text_with_real_citations(monkeypatch) -> None:
    monkeypatch.setattr("src.config.PROMPT_VERSION", "v3")
    seen = {}

    def fake(prompt, system=None, schema=None):
        seen["schema"] = schema
        return LLMResult(text=_json_reply(answer="It is 37.7%.", sources=[1]), input_tokens=3, output_tokens=2, latency_s=0.1)

    monkeypatch.setattr("src.generator.generate", fake)
    answer = answer_with_context("q?", [_hit(1, "a widget fact"), _hit(2, "other", source="b.pdf", page=4)])

    assert seen["schema"] is not None  # structured output was requested
    assert answer.text == "It is 37.7%. [1]"
    assert answer.citations == [{"n": 1, "source": "doc.pdf", "page": 1}]


def test_v3_found_false_becomes_not_found(monkeypatch) -> None:
    monkeypatch.setattr("src.config.PROMPT_VERSION", "v3")
    monkeypatch.setattr(
        "src.generator.generate",
        lambda p, system=None, schema=None: LLMResult(text=_json_reply(found=False), input_tokens=1, output_tokens=1, latency_s=0.1),
    )
    answer = answer_with_context("q?", [_hit(1, "a widget fact")])
    assert answer.text == "NOT_FOUND" and answer.citations == []


def test_v3_falls_back_to_free_text_on_malformed_json(monkeypatch) -> None:
    monkeypatch.setattr("src.config.PROMPT_VERSION", "v3")
    replies = iter(["this is not json", "Plain answer [1]."])
    monkeypatch.setattr(
        "src.generator.generate",
        lambda p, system=None, schema=None: LLMResult(text=next(replies), input_tokens=4, output_tokens=1, latency_s=0.5),
    )
    answer = answer_with_context("q?", [_hit(1, "a widget fact")])
    assert answer.text == "Plain answer [1]."
    assert (answer.tokens_in, answer.latency_s) == (8, 1.0)  # both calls counted


def test_ungrounded_numbers_flags_figures_missing_from_context() -> None:
    from src.generator import ungrounded_numbers

    context = ["Codex-S solves 37.7% of problems. The dataset has 100, 000 problems."]
    assert ungrounded_numbers("It solves 37.7% [1].", context) == []
    assert ungrounded_numbers("It has 100,000 problems.", context) == []  # "100, 000" in the PDF text
    assert ungrounded_numbers("It solves 41.2% of problems.", context) == ["41.2"]
    assert ungrounded_numbers("There are 3 variations [2].", context) == []  # small ints / citation markers ignored


def test_grounding_check_retries_once_on_a_wrong_number(monkeypatch) -> None:
    monkeypatch.setattr("src.config.PROMPT_VERSION", "v1")
    monkeypatch.setattr("src.config.GROUNDING_CHECK", True)
    replies = iter(["It is 41.2% [1].", "It is 37.7% [1]."])
    prompts = []

    def fake(prompt, system=None, schema=None):
        prompts.append(prompt)
        return LLMResult(text=next(replies), input_tokens=5, output_tokens=2, latency_s=1.0)

    monkeypatch.setattr("src.generator.generate", fake)
    answer = answer_with_context("q?", [_hit(1, "Codex-S solves 37.7% of problems.")])

    assert answer.text == "It is 37.7% [1]." and answer.ungrounded == []
    assert len(prompts) == 2 and "must appear" in prompts[1]
    assert (answer.tokens_in, answer.latency_s) == (10, 2.0)


def test_grounding_check_is_off_by_default_and_never_runs_on_refusals(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("src.config.PROMPT_VERSION", "v1")
    monkeypatch.setattr("src.generator.generate", lambda p, system=None, schema=None: (calls.append(1), LLMResult(text="It is 99.9% [1].", input_tokens=1, output_tokens=1, latency_s=0.1))[1])
    answer_with_context("q?", [_hit(1, "no such number")])
    assert len(calls) == 1  # GROUNDING_CHECK defaults to off

    monkeypatch.setattr("src.config.GROUNDING_CHECK", True)
    calls.clear()
    monkeypatch.setattr("src.generator.generate", lambda p, system=None, schema=None: (calls.append(1), LLMResult(text="NOT_FOUND", input_tokens=1, output_tokens=1, latency_s=0.1))[1])
    answer_with_context("q?", [_hit(1, "x")])
    assert len(calls) == 1
