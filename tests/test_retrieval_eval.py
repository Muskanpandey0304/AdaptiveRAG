"""Tests for src/retrieval_eval.py's pure helpers (retrieval itself needs the real index)."""

from src.retrieval_eval import describe, gold_hit
from src.retrievers import Hit


def _hit(source: str, page: int) -> Hit:
    return Hit(chunk_id=0, score=1.0, chunk={"source": source, "page": page})


def test_gold_hit_needs_both_source_and_page() -> None:
    question = {"gold_source": "a.pdf", "gold_pages": [2, 3]}
    assert gold_hit([_hit("b.pdf", 2), _hit("a.pdf", 3)], question)
    assert not gold_hit([_hit("b.pdf", 2), _hit("a.pdf", 9)], question)
    assert not gold_hit([], question)


def test_describe_reports_percentiles_and_handles_empty() -> None:
    assert describe([]) == "n/a"
    text = describe([0.0, 1.0])
    assert "min 0.000" in text and "max 1.000" in text and "median 0.500" in text
