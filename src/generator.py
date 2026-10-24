"""Turn retrieved chunks (or nothing) into a final answer with citations."""

import re
from dataclasses import dataclass

from src.llm import generate
from src.retrievers import Hit

_CITATION_RE = re.compile(r"\[(\d+)\]")

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
    result = generate(prompt, system=_CONTEXT_SYSTEM_PROMPT)

    citations = [] if result.text.strip() == "NOT_FOUND" else _parse_citations(result.text, hits)
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
