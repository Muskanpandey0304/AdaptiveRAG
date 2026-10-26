"""Load PDFs, clean their text, and split them into retrieval chunks.

Run as a script (`python -m src.ingest`) to read every PDF in data/raw/
and write data/index/chunks.json.
"""

import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from src.config import CHILD_OVERLAP, CHILD_TOKENS, CHUNK_OVERLAP, CHUNK_TOKENS

RAW_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/index/chunks.json")  # child chunks (what gets indexed)
PARENTS_PATH = Path("data/index/parents.json")  # parent chunks (what the LLM reads)

# Approximate a token as 1.3x a whitespace-split word, per the spec -- good
# enough for sizing chunks; we don't need an exact tokenizer here.
WORDS_TO_TOKENS = 1.3

# Not whitespace, so it survives the `\s+` collapse below while standing in
# for a paragraph break (a blank line) that we want chunk_pages to still be
# able to split on.
_PARA_MARKER = "\x00"


def load_pdfs(folder: str) -> list[dict]:
    """Read every PDF in `folder` into one dict per page.

    Returns a list of {"source": filename, "page": 1-indexed int, "text": str},
    in filename then page order.
    """
    pages: list[dict] = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            pages.append(
                {
                    "source": pdf_path.name,
                    "page": i + 1,
                    "text": page.extract_text() or "",
                }
            )
    return pages


def clean_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """Collapse whitespace, rejoin hyphenated line breaks, and drop repeated
    header/footer lines -- while keeping blank-line paragraph breaks so
    chunk_pages can still split on them.

    `repeated_lines` is computed once across a whole document's pages by
    `_find_repeated_lines`, since a single page can't tell what repeats.
    """
    if repeated_lines:
        lines = [line for line in text.split("\n") if line.strip() not in repeated_lines]
        text = "\n".join(lines)

    # A hyphen immediately followed by a line break is almost always a word
    # broken across the line, e.g. "imple-\nmentation" -> "implementation".
    text = re.sub(r"-\n", "", text)

    # A blank line (2+ consecutive newlines) is a paragraph break -- protect
    # it before the general whitespace collapse, then restore it afterwards.
    text = re.sub(r"\n{2,}", _PARA_MARKER, text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(_PARA_MARKER, "\n\n")
    return text.strip()


def _find_repeated_lines(pages: list[dict]) -> set[str]:
    """Find lines that appear on more than 50% of a document's pages.

    These are almost always running headers/footers (page numbers, titles)
    rather than content, so ingest drops them.
    """
    if not pages:
        return set()
    line_page_counts: Counter[str] = Counter()
    for page in pages:
        seen_this_page = {line.strip() for line in page["text"].split("\n") if line.strip()}
        line_page_counts.update(seen_this_page)
    threshold = len(pages) * 0.5
    return {line for line, count in line_page_counts.items() if count > threshold}


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    if not text:
        return []
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _approx_tokens(text: str) -> float:
    """Approximate token count as word count * 1.3, per the spec."""
    return len(text.split()) * WORDS_TO_TOKENS


def _units_for_paragraph(paragraph: str, max_tokens: float) -> list[str]:
    """Split one paragraph into pieces no bigger than ~max_tokens each.

    Tries to keep the whole paragraph as one unit; falls back to sentences,
    and -- only for the rare sentence that's still too big on its own --
    fixed-size word groups. This guarantees every unit chunk_pages packs is
    bounded, so a single run-on sentence can't blow a chunk past the
    CHUNK_TOKENS * 1.5 limit the acceptance test checks.
    """
    if _approx_tokens(paragraph) <= max_tokens:
        return [paragraph]

    units: list[str] = []
    for sentence in _split_sentences(paragraph):
        if _approx_tokens(sentence) <= max_tokens:
            units.append(sentence)
        else:
            words = sentence.split()
            step = max(1, int(max_tokens / WORDS_TO_TOKENS))
            for i in range(0, len(words), step):
                units.append(" ".join(words[i : i + step]))
    return units


def chunk_pages(
    pages: list[dict], chunk_tokens: int = CHUNK_TOKENS, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Split each page's text into ~chunk_tokens-token chunks with overlap.

    Splits on paragraph boundaries first, then sentence boundaries (see
    `_units_for_paragraph`). Chunks never cross a page boundary, since a
    citation needs to point at exactly one page. Each chunk:
    {id: int, text, source, page}.
    """
    chunks: list[dict] = []
    next_id = 0
    overlap_words = int(chunk_overlap / WORDS_TO_TOKENS)

    for page in pages:
        paragraphs = [p for p in page["text"].split("\n\n") if p.strip()]
        if not paragraphs and page["text"].strip():
            paragraphs = [page["text"]]

        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(_units_for_paragraph(paragraph, chunk_tokens))

        current_words: list[str] = []
        for unit in units:
            unit_words = unit.split()
            if current_words and _approx_tokens(" ".join(current_words)) + _approx_tokens(unit) > chunk_tokens:
                chunks.append(
                    {"id": next_id, "text": " ".join(current_words), "source": page["source"], "page": page["page"]}
                )
                next_id += 1
                current_words = current_words[-overlap_words:] if overlap_words else []
            current_words.extend(unit_words)

        if current_words:
            chunks.append(
                {"id": next_id, "text": " ".join(current_words), "source": page["source"], "page": page["page"]}
            )
            next_id += 1

    return chunks


def make_children(parents: list[dict]) -> list[dict]:
    """Split each parent chunk into small child chunks for indexing.

    Each child is {id, text, source, page, parent_id}; ids are sequential
    across all children (0..n-1, the same id==index-position invariant the
    Retriever relies on) and `parent_id` is the parent's own id. A parent
    never spans pages, so neither does any child -- a child's citation
    page is always its parent's page. Children are retrieved; the parent's
    larger text is what gets sent to the LLM.
    """
    children: list[dict] = []
    for parent in parents:
        page = {"source": parent["source"], "page": parent["page"], "text": parent["text"]}
        for child in chunk_pages([page], chunk_tokens=CHILD_TOKENS, chunk_overlap=CHILD_OVERLAP):
            children.append({**child, "id": len(children), "parent_id": parent["id"]})
    return children


def main() -> None:
    """CLI entry point: ingest data/raw/*.pdf into parents.json + chunks.json."""
    pages = load_pdfs(str(RAW_DIR))
    if not pages:
        print(f"No PDFs found in {RAW_DIR}/. Add PDF files there and re-run.")
        return

    repeated_lines = _find_repeated_lines(pages)
    for page in pages:
        page["text"] = clean_text(page["text"], repeated_lines)

    parents = chunk_pages(pages)
    children = make_children(parents)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(children, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(parents)} parents -> {PARENTS_PATH} and {len(children)} children -> {CHUNKS_PATH} "
        f"from {len(pages)} pages ({len(set(p['source'] for p in pages))} file(s))"
    )


if __name__ == "__main__":
    main()
