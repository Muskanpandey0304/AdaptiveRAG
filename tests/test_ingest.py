"""Tests for src/ingest.py.

chunk_pages and load_markdown are exercised with small synthetic pages and
Markdown strings -- this tests the chunking *logic*, not the project's real
documents.
"""

import random

from src.config import CHUNK_TOKENS
from src.ingest import chunk_pages, load_markdown, make_children


def _page(source: str, page: int, n_sentences: int) -> dict:
    """Build one page of `n_sentences` short, distinct sentences."""
    text = " ".join(f"This is sentence number {i} about widgets." for i in range(n_sentences))
    return {"source": source, "page": page, "text": text}


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


def test_print_five_random_chunks(capsys) -> None:
    """No assertion -- prints sample chunks for the user to eyeball, per
    the spec's Step 3 acceptance criterion."""
    chunks = chunk_pages([_page("doc.pdf", 1, 300)])
    for chunk in random.sample(chunks, min(5, len(chunks))):
        print(f"\n--- chunk {chunk['id']} (source={chunk['source']}, page={chunk['page']}) ---")
        print(chunk["text"][:300])


def test_make_children_link_to_parents_and_stay_on_the_same_page() -> None:
    parents = chunk_pages([_page("doc.pdf", 1, 300), _page("doc.pdf", 2, 300)])
    children = make_children(parents)

    assert [c["id"] for c in children] == list(range(len(children)))
    assert len(children) > len(parents)  # children are smaller, so more of them
    parent_by_id = {p["id"]: p for p in parents}
    for child in children:
        parent = parent_by_id[child["parent_id"]]
        assert (child["source"], child["page"]) == (parent["source"], parent["page"])
        assert child["text"] in parent["text"]  # a child is a span of its parent
    assert {c["parent_id"] for c in children} == set(parent_by_id)  # no parent left without children


def test_load_markdown_splits_pages_and_lifts_images_out_of_the_text(tmp_path) -> None:
    (tmp_path / "paper.md").write_text(
        "<!-- page: 1 -->\n\nFirst page text.\n\n![Image](paper_assets/fig1.png)\n\n"
        "<!-- page: 2 -->\n\n<!-- image -->\n\nSecond page text.\n",
        encoding="utf-8",
    )
    pages = load_markdown(tmp_path)
    assert [(p["source"], p["page"]) for p in pages] == [("paper.pdf", 1), ("paper.pdf", 2)]
    assert pages[0]["text"] == "First page text."
    assert pages[0]["images"] == ["paper_assets/fig1.png"]
    assert pages[1]["text"] == "Second page text."
    assert "images" not in pages[1]


def test_load_markdown_empty_folder(tmp_path) -> None:
    assert load_markdown(tmp_path) == []


def test_small_table_stays_whole_with_its_row_structure() -> None:
    table = "| Model | Acc |\n|---|---|\n| A | 1.0 |\n| B | 2.0 |"
    text = "Intro sentence about models.\n\n" + table + "\n\nAfter the table."
    chunks = chunk_pages([{"source": "d.pdf", "page": 1, "text": text}])
    assert len(chunks) == 1
    assert table in chunks[0]["text"]  # newlines preserved, not flattened into one line


def test_oversized_table_splits_by_rows_and_repeats_the_header() -> None:
    header = "| Model | Score |\n|---|---|"
    rows = [f"| model number {i} with a long descriptive name | {i}.5 |" for i in range(200)]
    page = {"source": "d.pdf", "page": 1, "text": header + "\n" + "\n".join(rows)}
    chunks = chunk_pages([page], chunk_tokens=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["text"].startswith(header)
        assert all(line.startswith("|") for line in chunk["text"].split("\n"))
    all_rows = [line for c in chunks for line in c["text"].split("\n")[2:]]
    assert all_rows == rows  # no row lost or duplicated


def test_display_formula_is_never_split() -> None:
    formula = "$$\nE = mc^2 + \\frac{a}{b}\n$$"
    filler = " ".join(f"Filler sentence {i} here." for i in range(60))
    page = {"source": "d.pdf", "page": 1, "text": f"{filler}\n\n{formula}\n\n{filler}"}
    chunks = chunk_pages([page], chunk_tokens=100)
    assert sum(formula in c["text"] for c in chunks) == 1


def test_heading_is_not_stranded_at_the_end_of_a_chunk() -> None:
    body = " ".join(f"Body sentence {i} about widgets." for i in range(16))  # about fills a 120-token chunk
    page = {"source": "d.pdf", "page": 1, "text": f"{body}\n\n## Results\n\n{body}"}
    chunks = chunk_pages([page], chunk_tokens=120, chunk_overlap=0)
    assert len(chunks) > 1
    assert not any(c["text"].rstrip().endswith("## Results") for c in chunks)
    assert any(c["text"].startswith("## Results") for c in chunks)


def test_chunks_carry_the_pages_images() -> None:
    page = {"source": "d.pdf", "page": 1, "text": "Some text.", "images": ["d_assets/a.png"]}
    assert chunk_pages([page])[0]["images"] == ["d_assets/a.png"]


def test_load_markdown_strips_table_padding(tmp_path) -> None:
    (tmp_path / "p.md").write_text(
        "<!-- page: 1 -->\n\n|    Model    |   Acc |\n|----------|-------|\n| A           |  0.9 |\n",
        encoding="utf-8",
    )
    assert load_markdown(tmp_path)[0]["text"] == "| Model | Acc |\n|---|---|\n| A | 0.9 |"


def test_paper_title_is_first_heading_with_stem_fallback() -> None:
    from src.ingest import paper_title

    assert paper_title("<!-- page: 1 -->\n\n## My Paper: A Study\n\n## Abstract\n", "x") == "My Paper: A Study"
    assert paper_title("no headings here", "stem") == "stem"


def test_chunks_and_children_carry_the_paper_title_and_contextualize_uses_it(tmp_path) -> None:
    from src.ingest import contextualize, make_children

    (tmp_path / "p.md").write_text("<!-- page: 1 -->\n\n## The Title\n\nSome body text here.\n", encoding="utf-8")
    parents = chunk_pages(load_markdown(tmp_path))
    children = make_children(parents)
    assert parents[0]["title"] == children[0]["title"] == "The Title"
    assert contextualize(children[0]).startswith("The Title\n")
    assert contextualize({"text": "bare"}) == "bare"  # no title -> unchanged
