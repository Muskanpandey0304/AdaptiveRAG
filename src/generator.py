"""Turn retrieved chunks (or nothing) into a final answer with citations."""

import re
from dataclasses import dataclass

from src.config import PROMPT_VERSION
from src.llm import generate
from src.retrievers import Hit

_CITATION_RE = re.compile(r"\[(\d+)\]")
# What's left of a reply once citation markers, digits and spacing are
# stripped -- a small model sometimes emits "[1] [2] [3] NOT_FOUND".
_NOT_FOUND_NOISE_RE = re.compile(r"[\[\]\d\s.,]")


def has_answer_text(text: str) -> bool:
    """True if `text` contains real words, not just citation markers like "[1] [2]"."""
    return len(re.sub(r"[^A-Za-z]", "", text)) >= 3


def is_not_found(text: str) -> bool:
    """True if `text` is the NOT_FOUND sentinel, even with stray citation
    markers or punctuation around it."""
    return _NOT_FOUND_NOISE_RE.sub("", text) == "NOT_FOUND"

_CONTEXT_SYSTEM_PROMPT = (
    "Answer the question using ONLY the numbered context blocks below. "
    "Cite every claim you make with the matching [n]. "
    "If the context does not contain the answer, reply with exactly: NOT_FOUND"
)


@dataclass
class Answer:
    """One generated answer plus everything needed to log/display it."""

    text: str
    citations: list[dict]
    tokens_in: int
    tokens_out: int
    latency_s: float

_CONTEXT_SYSTEM_PROMPT_V2 = (
    "Answer the question using ONLY the numbered context blocks below. "
    "Write your answer as one or more complete sentences that state the specific "
    "facts asked for (numbers, names, reasons) in your own words. "
    "After each claim, add the number of the supporting context block in square "
    "brackets, like [2]. Never reply with only a citation or only copied text. "
    "If the context does not contain the answer, reply with exactly: NOT_FOUND"
)

_RETRY_SUFFIX = (
    "\n\nYour previous reply contained no answer text. Answer the question now in "
    "complete sentences, citing blocks like [2]."
)


def _build_context_prompt(query: str, hits: list[Hit]) -> str:
    """Render `hits` as numbered context blocks [1], [2], ... for the LLM."""
    blocks = [f"[{i + 1}] {hit.chunk['text']}" for i, hit in enumerate(hits)]
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def _parse_citations(text: str, hits: list[Hit]) -> list[dict]:
    """Find every [n] in `text` and map it to {n, source, page}.

    A citation number that doesn't correspond to a context block (out of
    range, or the model invented one) is ignored rather than raised on --
    one bad citation shouldn't take down an otherwise-good answer.
    Duplicate citation numbers are only reported once, in first-seen order.
    """
    citations = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        n = int(match.group(1))
        if n in seen or not (1 <= n <= len(hits)):
            continue
        hit = hits[n - 1]
        citations.append({"n": n, "source": hit.chunk["source"], "page": hit.chunk["page"]})
        seen.add(n)
    return citations


def answer_with_context(query: str, hits: list[Hit]) -> Answer:
    """Answer `query` using only the given chunks as context, with citations."""
    prompt = _build_context_prompt(query, hits)
    system = _CONTEXT_SYSTEM_PROMPT_V2 if PROMPT_VERSION == "v2" else _CONTEXT_SYSTEM_PROMPT
    result = generate(prompt, system=system)

    # "v1r" = original prompt + the retry; "v2" = new prompt + the retry. A
    # citation-only reply is never a correct answer, so retrying it can't cost
    # accuracy, and it leaves the original prompt's refusal behavior alone.
    if PROMPT_VERSION in ("v1r", "v2") and not has_answer_text(result.text) and not is_not_found(result.text):
        first = result
        result = generate(prompt + _RETRY_SUFFIX, system=system)
        # Report the true cost of getting this answer, both calls.
        result = type(result)(
            text=result.text,
            input_tokens=first.input_tokens + result.input_tokens,
            output_tokens=first.output_tokens + result.output_tokens,
            latency_s=first.latency_s + result.latency_s,
        )

    citations = [] if is_not_found(result.text) else _parse_citations(result.text, hits)
    return Answer(
        text=result.text,
        citations=citations,
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        latency_s=result.latency_s,
    )


def answer_direct(query: str) -> Answer:
    """Answer `query` with no retrieved context and no citations."""
    result = generate(query)
    return Answer(
        text=result.text,
        citations=[],
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        latency_s=result.latency_s,
    )
