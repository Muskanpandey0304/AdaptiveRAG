"""Tests for src/router.py: route() classification for 28 labeled queries.

Pure regex/logic checks -- no retrieval, no LLM -- derived from the spec's
rule order (direct, then keyword, else hybrid). These labels are meant to
be reviewed by the user before being treated as the acceptance gate for
this step, per the spec.
"""

import pytest

from src.router import route

LABELED_QUERIES = [
    # --- direct ---
    ("Hi there!", "direct", "greeting_or_thanks"),
    ("Thanks a lot!", "direct", "greeting_or_thanks"),
    ("2 + 2", "direct", "pure_arithmetic"),
    ("15 * 3 - 7", "direct", "pure_arithmetic"),
    ("Who are you?", "direct", "identity_question"),
    ("What can you do?", "direct", "identity_question"),
    # --- keyword ---
    ('Search for "rate limiting" in the docs', "keyword", "quoted_phrase"),
    ("What does the error 'connection refused' mean", "keyword", "quoted_phrase"),
    ("JIRA-4532 status", "keyword", "identifier_pattern"),
    ("AB1234 error code", "keyword", "identifier_pattern"),
    ("What changed in v2.1.0?", "keyword", "version_pattern"),
    ("Release 1.4 notes", "keyword", "version_pattern"),
    ("explain chunk_pages function", "keyword", "snake_case_token"),
    ("build_vector_index parameters", "keyword", "snake_case_token"),
    ("getUserById implementation", "keyword", "camel_case_token"),
    ("loadPdfs return type", "keyword", "camel_case_token"),
    ("open src/config.py", "keyword", "file_path"),
    ("check data/raw/notes.txt", "keyword", "file_path"),
    ("--max-findings flag usage", "keyword", "cli_flag"),
    ("pass --no-verify carefully", "keyword", "cli_flag"),
    ("database connection pooling", "keyword", "short_query_no_question_word"),
    ("token usage limits", "keyword", "short_query_no_question_word"),
    # --- hybrid ---
    (
        "Why might two different retrieval methods agree on an answer but disagree on which pages support it?",
        "hybrid",
        "default",
    ),
    ("Why would a chunk's confidence score be low even if it looks relevant?", "hybrid", "default"),
    ("Explain the tradeoffs between keyword and semantic search in this project", "hybrid", "default"),
    (
        "What are the differences between RRF fusion and simple score averaging for combining retrieval results?",
        "hybrid",
        "default",
    ),
    ("When should I use reranking instead of just taking the top vector search hits?", "hybrid", "default"),
    ("Can you summarize how confidence thresholds were chosen for this pipeline?", "hybrid", "default"),
]


@pytest.mark.parametrize("query,expected_route,expected_reason", LABELED_QUERIES)
def test_route_classification(query: str, expected_route: str, expected_reason: str) -> None:
    actual_route, actual_reason = route(query)
    assert (actual_route, actual_reason) == (expected_route, expected_reason), (
        f"query={query!r} -> got ({actual_route!r}, {actual_reason!r}), "
        f"expected ({expected_route!r}, {expected_reason!r})"
    )
