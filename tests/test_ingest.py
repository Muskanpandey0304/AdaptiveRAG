"""Tests for src/ingest.py.

chunk_pages and clean_text are exercised with small synthetic page dicts --
this tests the chunking/cleaning *logic*, not the project's real document
data. data/raw/ is intentionally empty (the user hasn't supplied PDFs yet);
per the project's rule against inventing data, we don't fabricate fake PDFs
to stand in for the real corpus.
"""

import random

from src.config import CHUNK_TOKENS
from src.ingest import _find_repeated_lines, chunk_pages, clean_text, load_pdfs


def _page(source: str, page: int, n_sentences: int) -> dict:
    """Build one page of `n_sentences` short, distinct sentences."""
    text = " ".join(f"This is sentence number {i} about widgets." for i in range(n_sentences))
    return {"source": source, "page": page, "text": text}


def test_load_pdfs_returns_empty_list_for_empty_folder(tmp_path) -> None:
    assert load_pdfs(str(tmp_path)) == []


def test_chunks_have_all_fields_and_are_nonempty() -> None:
    chunks = chunk_pages([_page("doc.pdf", 1, 200)])
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["text"].strip() != ""
        assert set(chunk.keys()) == {"id", "text", "source", "page"}


def test_chunk_ids_unique_and_sequential() -> None:
    chunks = chunk_pages([_page("doc.pdf", 1, 200), _page("doc.pdf", 2, 200)])
    ids = [c["id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_no_chunk_exceeds_token_limit() -> None:
    chunks = chunk_pages([_page("doc.pdf", 1, 500)])
    max_tokens = CHUNK_TOKENS * 1.5
    for chunk in chunks:
        approx_tokens = len(chunk["text"].split()) * 1.3
        assert approx_tokens <= max_tokens


def test_chunks_never_cross_page_boundary() -> None:
    chunks = chunk_pages([_page("doc.pdf", 1, 50), _page("doc.pdf", 2, 50)])
    assert {c["page"] for c in chunks} == {1, 2}


def test_clean_text_dehyphenates_and_collapses_whitespace() -> None:
    raw = "This is an imple-\nmentation   detail.\n\nNext paragraph."
    cleaned = clean_text(raw)
    assert "imple-\nmentation" not in cleaned
    assert "implementation" in cleaned
    assert "  " not in cleaned


def test_clean_text_keeps_paragraph_breaks_for_chunking() -> None:
    raw = "First paragraph here.\n\nSecond paragraph here."
    cleaned = clean_text(raw)
    assert "\n\n" in cleaned


def test_clean_text_drops_repeated_header_footer_lines() -> None:
    # The header/footer repeats on every page; the body differs per page so
    # it's only the header/footer that crosses the 50%-of-pages threshold.
    pages = [
        {"source": "doc.pdf", "page": i, "text": f"CONFIDENTIAL\nReal content for page {i}.\nCONFIDENTIAL"}
        for i in range(1, 5)
    ]
    repeated = _find_repeated_lines(pages)
    assert "CONFIDENTIAL" in repeated

    cleaned = clean_text(pages[0]["text"], repeated)
    assert "CONFIDENTIAL" not in cleaned
    assert "Real content for page 1." in cleaned


def test_print_five_random_chunks(capsys) -> None:
    """No assertion -- prints sample chunks for the user to eyeball, per
    the spec's Step 3 acceptance criterion."""
    chunks = chunk_pages([_page("doc.pdf", 1, 300)])
    for chunk in random.sample(chunks, min(5, len(chunks))):
        print(f"\n--- chunk {chunk['id']} (source={chunk['source']}, page={chunk['page']}) ---")
        print(chunk["text"][:300])
