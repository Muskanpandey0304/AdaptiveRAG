"""Tests for src/llm.py.

Ollama needs no API key, so the thing that makes this test unrunnable in CI
or on a fresh machine is the local Ollama server not being up (or not having
the configured model pulled) rather than a missing key. We skip in that case
instead of failing, mirroring the spec's "skipped if no key" intent.
"""

import ollama
import pytest

from src.config import LLM_MODEL
from src.llm import generate


def _ollama_available() -> bool:
    """True only if a local Ollama server responds and has the configured model pulled."""
    try:
        pulled = [m.model for m in ollama.list().models]
    except Exception:
        return False
    wanted = LLM_MODEL if ":" in LLM_MODEL else f"{LLM_MODEL}:latest"
    return wanted in pulled


@pytest.mark.skipif(not _ollama_available(), reason="no local Ollama server, or the configured model is not pulled")
def test_generate_returns_nonempty_text() -> None:
    """A basic call should succeed and return a non-empty answer."""
    result = generate("Say OK")
    assert result.text.strip() != ""


class _Msg:
    content = "ok"


class _Resp:
    message = _Msg()
    prompt_eval_count = 1
    eval_count = 1


def _fake_client(monkeypatch, chat):
    """Make src.llm build a fake ollama.Client whose chat() is `chat`."""
    import src.llm as llm_module

    created = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def chat(self, model, messages, **kwargs):
            return chat(model, messages, **kwargs)

    monkeypatch.setattr(llm_module.ollama, "Client", FakeClient)
    return created


def test_ollama_call_caps_output_pins_context_and_is_deterministic(monkeypatch) -> None:
    """Regression: with no num_predict, a looping model produced a ~209K-char answer.
    num_ctx is pinned because Ollama silently truncates prompts past its default window."""
    import src.llm as llm_module
    from src.config import LLM_KEEP_ALIVE, LLM_MAX_TOKENS, LLM_NUM_CTX, LLM_TEMPERATURE, LLM_TIMEOUT_S

    seen = {}

    def chat(model, messages, **kwargs):
        seen.update(kwargs)
        return _Resp()

    created = _fake_client(monkeypatch, chat)
    llm_module.generate("hi")
    assert seen["options"] == {"num_predict": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX}
    assert seen["keep_alive"] == LLM_KEEP_ALIVE
    assert created["kwargs"] == {"timeout": LLM_TIMEOUT_S}


def test_ollama_call_passes_a_json_schema_only_when_given(monkeypatch) -> None:
    import src.llm as llm_module

    seen = []

    def chat(model, messages, **kwargs):
        seen.append(kwargs)
        return _Resp()

    _fake_client(monkeypatch, chat)
    llm_module.generate("hi")
    llm_module.generate("hi", schema={"type": "object"})

    assert "format" not in seen[0]
    assert seen[1]["format"] == {"type": "object"}


def test_transient_errors_are_retried_once(monkeypatch) -> None:
    import httpx

    import src.llm as llm_module

    for error in (ConnectionError("down"), httpx.ConnectError("refused"), httpx.ReadTimeout("slow")):
        calls = []

        def chat(model, messages, _error=error, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise _error
            return _Resp()

        _fake_client(monkeypatch, chat)
        assert llm_module.generate("hi").text == "ok"
        assert len(calls) == 2, error


def test_non_transient_errors_are_not_retried(monkeypatch) -> None:
    import ollama

    import src.llm as llm_module

    calls = []

    def chat(model, messages, **kwargs):
        calls.append(1)
        raise ollama.ResponseError("model 'nope' not found", 404)

    _fake_client(monkeypatch, chat)
    with pytest.raises(ollama.ResponseError):
        llm_module.generate("hi")
    assert len(calls) == 1


def test_a_second_transient_failure_propagates(monkeypatch) -> None:
    import src.llm as llm_module

    calls = []

    def chat(model, messages, **kwargs):
        calls.append(1)
        raise ConnectionError("still down")

    _fake_client(monkeypatch, chat)
    with pytest.raises(ConnectionError):
        llm_module.generate("hi")
    assert len(calls) == 2
