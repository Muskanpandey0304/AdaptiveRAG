"""Turn retrieved chunks (or nothing) into a final answer with citations."""

import json
import re
from dataclasses import dataclass, field

from src import config
from src.ingest import contextualize
from src.llm import LLMResult, generate
from src.retrievers import Hit

_CITATION_RE = re.compile(r"\[(\d+)\]")
# What's left of a reply once citation markers, digits and spacing are
# stripped -- a small model sometimes emits "[1] [2] [3] NOT_FOUND".
_NOT_FOUND_NOISE_RE = re.compile(r"[\[\]\d\s.,]")
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


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

# "v1c": v1 with ONE sentence changed. v1 says "the matching [n]", and the 3B
# model sometimes copies "[n]" literally (7 of 46 answers in one run), which
# leaves the answer with no parseable citation. "e.g. [2]" shows what a real
# marker looks like. Everything else, including the NOT_FOUND sentence, is
# word-for-word v1 so refusal behaviour is not disturbed (the fuller rewrite v2
# loosened refusals).
_CONTEXT_SYSTEM_PROMPT_V1C = (
    "Answer the question using ONLY the numbered context blocks below. "
    "After each claim, write the number of the supporting context block in square "
    "brackets, e.g. [2]. "
    "If the context does not contain the answer, reply with exactly: NOT_FOUND"
)

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

# "v3": the reply is forced into this JSON shape, so there is no free-text
# citation syntax to get wrong and a refusal is an explicit found=false.
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "sources"],
}

_STRUCTURED_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context blocks provided. "
    "Reply as JSON with three fields. "
    "found: true only if the context contains the answer, otherwise false. "
    "answer: if found, one or more complete sentences stating the specific facts "
    "asked for (numbers, names, reasons) in your own words, never copied text "
    "dumps; if not found, an empty string. "
    "sources: the numbers of the context blocks that support the answer."
)

_GROUNDING_SUFFIX = (
    "\n\nCareful: every number in your answer must appear in the context blocks. "
    "Re-read the blocks and answer again using only numbers that are written there."
)


@dataclass
class Answer:
    """One generated answer plus everything needed to log/display it."""

    text: str
    citations: list[dict]
    tokens_in: int
    tokens_out: int
    latency_s: float
    # Numbers in the answer that appear nowhere in the retrieved context
    # (only filled in when GROUNDING_CHECK is on); non-empty = suspect answer.
    ungrounded: list[str] = field(default_factory=list)


def _build_context_prompt(query: str, hits: list[Hit]) -> str:
    """Render `hits` as numbered context blocks [1], [2], ... for the LLM."""
    blocks = [f"[{i + 1}] {contextualize(hit.chunk)}" for i, hit in enumerate(hits)]
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


def _numbers(text: str) -> set[str]:
    """Normalized numbers in `text` ("37.7%" -> "37.7", "100, 000" -> "100000").

    Single-digit-or-ten integers are ignored: they're usually counts or list
    positions the model derives rather than copies, and would cause false alarms.
    """
    text = re.sub(r"(\d),\s+(\d)", r"\1\2", text)  # PDFs often write "100, 000"
    found = set()
    for match in _NUMBER_RE.finditer(text):
        token = match.group().replace(",", "").rstrip(".")
        if not token or (token.isdigit() and int(token) <= 10):
            continue
        found.add(token)
    return found


def ungrounded_numbers(answer: str, context_texts: list[str]) -> list[str]:
    """Numbers stated in `answer` that appear in none of `context_texts`."""
    answer = re.sub(r"\[[^\]]*\]", "", answer)  # drop citation markers
    context_numbers: set[str] = set()
    for text in context_texts:
        context_numbers |= _numbers(text)
    return sorted(_numbers(answer) - context_numbers)


def _sum_cost(first: LLMResult, second: LLMResult) -> LLMResult:
    """The true cost of getting an answer that took two calls."""
    return LLMResult(
        text=second.text,
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        latency_s=first.latency_s + second.latency_s,
    )


def _structured_reply(result: LLMResult) -> LLMResult | None:
    """Turn a "v3" JSON reply into display text ("answer [1][2]" or NOT_FOUND).

    Returns None if the reply isn't the expected JSON, so the caller can fall
    back to free text instead of failing.
    """
    try:
        data = json.loads(result.text)
        found, answer, sources = bool(data["found"]), str(data["answer"]).strip(), data["sources"]
    except (ValueError, KeyError, TypeError):
        return None
    if not found or not has_answer_text(answer):
        text = "NOT_FOUND"
    else:
        markers = "".join(f"[{n}]" for n in sources if isinstance(n, int))
        text = f"{answer} {markers}".strip()
    return LLMResult(text=text, input_tokens=result.input_tokens, output_tokens=result.output_tokens, latency_s=result.latency_s)


def _generate_text(prompt: str, version: str) -> LLMResult:
    """One generation in the style of `version` ("v1", "v1r", "v1c", "v2" or "v3")."""
    if version == "v3":
        raw = generate(prompt, system=_STRUCTURED_SYSTEM_PROMPT, schema=_ANSWER_SCHEMA)
        parsed = _structured_reply(raw)
        if parsed is not None:
            return parsed
        version = "v1r"  # malformed JSON: fall back to free text rather than fail
        raw_cost = raw
    else:
        raw_cost = None

    system = {"v2": _CONTEXT_SYSTEM_PROMPT_V2, "v1c": _CONTEXT_SYSTEM_PROMPT_V1C}.get(version, _CONTEXT_SYSTEM_PROMPT)
    result = generate(prompt, system=system)
    # "v1r" = original prompt + the retry; "v2" = new prompt + the retry. A
    # citation-only reply is never a correct answer, so retrying it can't cost
    # accuracy, and it leaves the original prompt's refusal behavior alone.
    if version in ("v1r", "v1c", "v2") and not has_answer_text(result.text) and not is_not_found(result.text):
        result = _sum_cost(result, generate(prompt + _RETRY_SUFFIX, system=system))
    return _sum_cost(raw_cost, result) if raw_cost is not None else result


def answer_with_context(query: str, hits: list[Hit], full_hits: list[Hit] | None = None) -> Answer:
    """Answer `query` using only the given chunks as context, with citations.

    `full_hits`, if given, are the uncompressed parents `hits` were derived
    from (CONTEXT_MODE="sentences"); the number check is made against them so
    an answer isn't flagged for a figure that sits just outside the excerpt.
    """
    version = config.PROMPT_VERSION
    prompt = _build_context_prompt(query, hits)
    result = _generate_text(prompt, version)

    ungrounded: list[str] = []
    if config.GROUNDING_CHECK and not is_not_found(result.text):
        context_texts = [h.chunk["text"] for h in (full_hits or hits)]
        ungrounded = ungrounded_numbers(result.text, context_texts)
        if ungrounded:
            # Retry once, on the fuller context when we have it.
            retry_hits = full_hits or hits
            retry = _generate_text(_build_context_prompt(query, retry_hits) + _GROUNDING_SUFFIX, version)
            result = _sum_cost(result, retry)
            hits = retry_hits
            ungrounded = [] if is_not_found(result.text) else ungrounded_numbers(result.text, context_texts)

    citations = [] if is_not_found(result.text) else _parse_citations(result.text, hits)
    return Answer(
        text=result.text,
        citations=citations,
        tokens_in=result.input_tokens,
        tokens_out=result.output_tokens,
        latency_s=result.latency_s,
        ungrounded=ungrounded,
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
