"""Ties the router, retrievers, and generator into one entry point: ask().

Logs one JSON line per call to logs/queries.jsonl so Step 8's eval can
inspect what actually happened (route, confidence, tokens...), not just
final answer text.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.config import (
    BM25_MIN_SCORE,
    CANDIDATES_K,
    CENTROID_MIN_SIM,
    RERANK_MIN_PROB,
    RERANK_POOL_K,
    TOP_K,
    VECTOR_MIN_SIM,
)
from src.generator import Answer, answer_direct, answer_with_context, is_not_found
from src.retrievers import Hit, Retriever
from src.router import is_factoid_question, route

LOG_PATH = Path("logs/queries.jsonl")
NOT_FOUND_TEXT = "Not found in the documents."

# Lazily constructed: building a Retriever needs real chunks.json/
# faiss.index/bm25.pkl on disk, which shouldn't be required just to import
# this module (e.g. to test the "direct" route, which never touches
# retrieval at all).
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@dataclass
class Result:
    """Everything about one ask() call: the answer plus how it got there."""

    answer: Answer
    route: str
    reason: str
    escalated: bool
    confidence: float | None
    centroid_sim: float | None
    latency_s: float
    tokens_in: int
    tokens_out: int
    hits: list[Hit]


def _not_found_answer() -> Answer:
    """A fixed non-answer for low-confidence misses -- no LLM call is made
    to produce it, so tokens/latency are all 0."""
    return Answer(text=NOT_FOUND_TEXT, citations=[], tokens_in=0, tokens_out=0, latency_s=0.0)


def _normalize_not_found(answer: Answer) -> Answer:
    """Map the generator's NOT_FOUND sentinel to the user-facing message.

    Keeps the real token/latency numbers from the LLM call that produced
    it, unlike the confidence-gated `_not_found_answer()` case above.
    """
    if not is_not_found(answer.text):
        return answer
    return Answer(text=NOT_FOUND_TEXT, citations=[], tokens_in=answer.tokens_in, tokens_out=answer.tokens_out, latency_s=answer.latency_s)


def _expand_if_enabled(hits: list[Hit], expand_neighbors: bool) -> list[Hit]:
    """Apply neighbor expansion (if enabled) as the last step before
    generation -- after retrieval/rerank and the confidence check, since a
    low-confidence miss shouldn't pay for it, and expanding is about
    giving the LLM fuller context for chunks already judged relevant, not
    about re-ranking candidates."""
    if not expand_neighbors or not hits:
        return hits
    return _get_retriever().expand_neighbors(hits, window=1)


def _generate_answer(query: str, hits: list[Hit]) -> Answer:
    """Generate from `hits`, shrinking them to their best sentences first when
    CONTEXT_MODE is "sentences". The full parents stay available to the
    generator's number check, and `result.hits` still reports the full parents."""
    llm_hits = hits
    use_sentences = config.CONTEXT_MODE == "sentences" or (
        config.CONTEXT_MODE == "auto" and is_factoid_question(query)
    )
    if use_sentences and hits:
        llm_hits = _get_retriever().select_sentences(query, hits)
    return _normalize_not_found(answer_with_context(query, llm_hits, full_hits=hits))


def _run_hybrid(query: str, use_rerank: bool, expand_neighbors: bool) -> tuple[Answer, float, float, list[Hit]]:
    """Hybrid retrieval + generation. Shared by the hybrid route and by
    keyword-route escalation, since escalation *is* "run the hybrid flow".

    Retrieval works on small child chunks; the reranker (if enabled) sees
    a wide RERANK_POOL_K pool of them -- wider than the TOP_K it keeps, or
    it would only reshuffle the same few hits -- and only then are children
    collapsed to the larger parent chunks the LLM reads.
    """
    retriever = _get_retriever()
    pool = retriever.hybrid_search(query, RERANK_POOL_K)
    if use_rerank and pool:
        pool = retriever.rerank(query, pool, len(pool))
        hits = retriever.to_parents(pool)[:TOP_K]
        confidence = hits[0].score  # sigmoid-normalized rerank probability
        min_confidence = RERANK_MIN_PROB
    else:
        hits = retriever.to_parents(pool)[:TOP_K]
        confidence = hits[0].meta.get("vector_sim", 0.0) if hits else 0.0
        min_confidence = VECTOR_MIN_SIM
    centroid_sim = retriever.centroid_similarity(query)

    if confidence < min_confidence or centroid_sim < CENTROID_MIN_SIM:
        return _not_found_answer(), confidence, centroid_sim, hits
    hits = _expand_if_enabled(hits, expand_neighbors)
    return _generate_answer(query, hits), confidence, centroid_sim, hits


def _run_vector(query: str, expand_neighbors: bool) -> tuple[Answer, float, float, list[Hit]]:
    """Plain vector search then generate -- the Step 6 baseline, reachable
    via force_route="vector" so eval can compare the router against it.
    Searches CANDIDATES_K children, so TOP_K distinct parents are likely."""
    retriever = _get_retriever()
    hits = retriever.to_parents(retriever.vector_search(query, CANDIDATES_K))[:TOP_K]
    confidence = hits[0].score if hits else 0.0
    centroid_sim = retriever.centroid_similarity(query)

    if confidence < VECTOR_MIN_SIM or centroid_sim < CENTROID_MIN_SIM:
        return _not_found_answer(), confidence, centroid_sim, hits
    hits = _expand_if_enabled(hits, expand_neighbors)
    return _generate_answer(query, hits), confidence, centroid_sim, hits


def _log(result: Result, query: str) -> None:
    """Append one JSON line describing this call to logs/queries.jsonl."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "query": query,
        "route": result.route,
        "reason": result.reason,
        "escalated": result.escalated,
        "confidence": result.confidence,
        "centroid_sim": result.centroid_sim,
        "latency_s": result.latency_s,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "top_chunk_ids": [hit.chunk_id for hit in result.hits],
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def ask(
    query: str,
    force_route: str | None = None,
    use_rerank: bool = False,
    expand_neighbors: bool = False,
) -> Result:
    """Answer `query`, routing it automatically unless `force_route` is set.

    `force_route` in {"direct", "keyword", "hybrid", "vector"} bypasses
    both the router and keyword-route escalation, so eval can measure each
    strategy in isolation. "vector" is the Step 6 plain-vector baseline --
    not a route the router itself ever picks, only reachable via this.

    `expand_neighbors`, if True, adds each retrieved chunk's immediately
    adjacent chunks (same source, see Retriever.expand_neighbors) right
    before generation, to fix answers split across a chunk boundary.
    Applied only on an actual answer attempt (not a confidence-gated
    miss), and `result.hits` reflects the expanded set, since that's what
    was actually available to the LLM.

    Every route that touches retrieval (keyword/vector/hybrid) also
    checks `Retriever.centroid_similarity(query)` against
    `CENTROID_MIN_SIM` before generating -- a corpus-level "is this query
    broadly like my documents at all" signal, on top of each method's own
    per-chunk confidence check. Added post-spec: see DECISIONS.md.
    """
    start = time.perf_counter()

    if force_route is not None:
        chosen_route, reason, escalated = force_route, "forced", False
    else:
        chosen_route, reason = route(query)
        escalated = False

    hits: list[Hit] = []
    confidence: float | None = None
    centroid_sim: float | None = None

    if chosen_route == "direct":
        answer = answer_direct(query)

    elif chosen_route == "keyword":
        retriever = _get_retriever()
        hits = retriever.to_parents(retriever.bm25_search(query, CANDIDATES_K))[:TOP_K]
        confidence = hits[0].score if hits else 0.0
        if force_route is None and confidence < BM25_MIN_SCORE:
            escalated = True
            answer, confidence, centroid_sim, hits = _run_hybrid(query, use_rerank, expand_neighbors)
        else:
            # BM25's literal keyword matching is exactly what let an
            # off-topic query match a corpus's own example Q&A/benchmark
            # text (e.g. "What is the capital of France?" matching a
            # paper's sample question) -- the centroid gate catches that
            # even when BM25's own score looks confident.
            centroid_sim = _get_retriever().centroid_similarity(query)
            if centroid_sim < CENTROID_MIN_SIM:
                answer = _not_found_answer()
            else:
                hits = _expand_if_enabled(hits, expand_neighbors)
                answer = _generate_answer(query, hits)

    elif chosen_route == "vector":
        answer, confidence, centroid_sim, hits = _run_vector(query, expand_neighbors)

    else:  # "hybrid"
        answer, confidence, centroid_sim, hits = _run_hybrid(query, use_rerank, expand_neighbors)

    latency_s = time.perf_counter() - start
    result = Result(
        answer=answer,
        route=chosen_route,
        reason=reason,
        escalated=escalated,
        confidence=confidence,
        centroid_sim=centroid_sim,
        latency_s=latency_s,
        tokens_in=answer.tokens_in,
        tokens_out=answer.tokens_out,
        hits=hits,
    )
    _log(result, query)
    return result
