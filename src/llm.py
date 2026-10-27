"""Thin wrapper around the configured LLM provider.

The rest of the pipeline only ever calls `generate()`. Only the "ollama"
provider is implemented here (per the project's LLM_PROVIDER choice and the
spec's rule to implement just the chosen provider); any other value is
rejected loudly instead of silently doing nothing.
"""

import time
from dataclasses import dataclass

import httpx
import ollama

from src.config import (
    LLM_KEEP_ALIVE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
)

# Errors worth one retry: the server dropped the connection or was slow while
# loading the model. ollama's client raises the builtin ConnectionError when
# it can't reach the server and passes httpx timeouts through.
_TRANSIENT_ERRORS = (ConnectionError, httpx.ConnectError, httpx.TimeoutException)


@dataclass
class LLMResult:
    """Everything the rest of the pipeline needs from one LLM call."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def generate(prompt: str, system: str | None = None, schema: dict | None = None) -> LLMResult:
    """Call the configured LLM once and return its answer plus call stats.

    `schema`, if given, is a JSON schema the reply must follow (Ollama's
    structured output); the reply text is then a JSON string."""
    if LLM_PROVIDER != "ollama":
        raise NotImplementedError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} is not implemented; only 'ollama' is."
        )
    return _generate_ollama(prompt, system, schema)


def _generate_ollama(prompt: str, system: str | None, schema: dict | None = None) -> LLMResult:
    """Send `prompt` to the local Ollama server and time the round trip.

    Retries once, only on connection errors and timeouts: a local Ollama
    server occasionally drops the first request to a model while it's still
    being loaded into memory, which is transient rather than a real failure.
    """
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "options": {"num_predict": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX},
        "keep_alive": LLM_KEEP_ALIVE,
    }
    if schema is not None:
        kwargs["format"] = schema

    client = ollama.Client(timeout=LLM_TIMEOUT_S)
    try:
        start = time.perf_counter()
        response = client.chat(model=LLM_MODEL, messages=messages, **kwargs)
    except _TRANSIENT_ERRORS:
        # Retry only connection hiccups and timeouts. Anything else (e.g. the
        # model isn't pulled) would fail identically, so it propagates at once.
        # Latency is timed per attempt, so a failed first try doesn't inflate it.
        start = time.perf_counter()
        response = client.chat(model=LLM_MODEL, messages=messages, **kwargs)
    latency_s = time.perf_counter() - start

    return LLMResult(
        text=response.message.content or "",
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        latency_s=latency_s,
    )
