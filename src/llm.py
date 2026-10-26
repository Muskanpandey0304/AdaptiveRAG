"""Thin wrapper around the configured LLM provider.

The rest of the pipeline only ever calls `generate()`. Only the "ollama"
provider is implemented here (per the project's LLM_PROVIDER choice and the
spec's rule to implement just the chosen provider); any other value is
rejected loudly instead of silently doing nothing.
"""

import time
from dataclasses import dataclass

import ollama

from src.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE


@dataclass
class LLMResult:
    """Everything the rest of the pipeline needs from one LLM call."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def generate(prompt: str, system: str | None = None) -> LLMResult:
    """Call the configured LLM once and return its answer plus call stats."""
    if LLM_PROVIDER != "ollama":
        raise NotImplementedError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} is not implemented; only 'ollama' is."
        )
    return _generate_ollama(prompt, system)


def _generate_ollama(prompt: str, system: str | None) -> LLMResult:
    """Send `prompt` to the local Ollama server and time the round trip.

    Retries once: a local Ollama server occasionally drops the first request
    to a model while it's still being loaded into memory, which is
    transient rather than a real failure.
    """
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"num_predict": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE}

    start = time.perf_counter()
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages, options=options)
    except Exception:
        response = ollama.chat(model=LLM_MODEL, messages=messages, options=options)
    latency_s = time.perf_counter() - start

    return LLMResult(
        text=response.message.content or "",
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        latency_s=latency_s,
    )
