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
