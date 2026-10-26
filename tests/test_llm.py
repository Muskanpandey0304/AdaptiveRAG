"""Tests for src/llm.py.

Ollama needs no API key, so the thing that makes this test unrunnable in CI
or on a fresh machine is the local Ollama server not being up (or not having
the configured model pulled) rather than a missing key. We skip in that case
instead of failing, mirroring the spec's "skipped if no key" intent.
"""

import ollama
import pytest

from src.llm import generate


def _ollama_available() -> bool:
    """Return True only if a local Ollama server responds."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="no local Ollama server running")
def test_generate_returns_nonempty_text() -> None:
    """A basic call should succeed and return a non-empty answer."""
    result = generate("Say OK")
    assert result.text.strip() != ""


def test_ollama_call_caps_output_and_is_deterministic(monkeypatch) -> None:
    """Regression: with no num_predict, a looping model produced a ~209K-char answer."""
    import src.llm as llm_module
    from src.config import LLM_MAX_TOKENS, LLM_TEMPERATURE

    seen = {}

    class _Msg:
        content = "ok"

    class _Resp:
        message = _Msg()
        prompt_eval_count = 1
        eval_count = 1

    def fake_chat(model, messages, options=None):
        seen["options"] = options
        return _Resp()

    monkeypatch.setattr(llm_module.ollama, "chat", fake_chat)
    llm_module.generate("hi")
    assert seen["options"] == {"num_predict": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE}


def test_ollama_call_passes_a_json_schema_only_when_given(monkeypatch) -> None:
    import src.llm as llm_module

    seen = []

    class _Msg:
        content = "{}"

    class _Resp:
        message = _Msg()
        prompt_eval_count = 1
        eval_count = 1

    def fake_chat(model, messages, **kwargs):
        seen.append(kwargs)
        return _Resp()

    monkeypatch.setattr(llm_module.ollama, "chat", fake_chat)
    llm_module.generate("hi")
    llm_module.generate("hi", schema={"type": "object"})

    assert "format" not in seen[0]
    assert seen[1]["format"] == {"type": "object"}
