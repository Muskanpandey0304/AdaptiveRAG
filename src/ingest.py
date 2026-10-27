"""Read the Markdown produced by `src.convert` and split it into retrieval chunks.

Run as a script (`python -m src.ingest`): it first converts any new or changed
PDF in data/raw/ to data/md/ (see src/convert.py), then chunks every .md file
and writes data/index/parents.json + chunks.json. Markdown, not the PDF, is the
single source of truth for chunking.

Markdown-aware chunking: a table or a display formula is never cut in half
(an oversized table is split by rows, repeating its header), and a heading is
never left stranded at the end of a chunk away from the text it introduces.
"""

import json
import re
from pathlib import Path

from src.config import CHILD_OVERLAP, CHILD_TOKENS, CHUNK_OVERLAP, CHUNK_TOKENS

MD_DIR = Path("data/md")
CHUNKS_PATH = Path("data/index/chunks.json")  # child chunks (what gets indexed)
PARENTS_PATH = Path("data/index/parents.json")  # parent chunks (what the LLM reads)

# Approximate a token as 1.3x a whitespace-split word, per the spec -- good
# enough for sizing chunks; we don't need an exact tokenizer here.
WORDS_TO_TOKENS = 1.3

_PAGE_RE = re.compile(r"^<!-- page: (\d+) -->[ \t]*$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)  # e.g. docling's "<!-- image -->" placeholder
_HEADING_RE = re.compile(r"#{1,6}\s")
_TABLE_PAD_RE = re.compile(r"(?m)^(\s*\|.*)$")

# Unit kinds: "prose" can be split and overlapped; "heading" must stay with the
# text after it; "atomic" (table piece, display formula) is never cut.
_PROSE, _HEADING, _ATOMIC = "prose", "heading", "atomic"

Unit = tuple[str, str, str]  # (text, kind, separator placed before it)


def _compact_tables(text: str) -> str:
    """Strip the column-alignment padding docling puts in pipe tables ("|  a   |"
    and "|-----|"). It carries no information but can multiply a wide table's
    size many times over, wasting chunk budget and LLM context."""

    def compact(match: re.Match) -> str:
        line = re.sub(r" {2,}", " ", match.group(1))
        return re.sub(r"-{4,}", "---", line)

    return _TABLE_PAD_RE.sub(compact, text)


def load_markdown(folder: Path | str = MD_DIR) -> list[dict]:
    """Read every .md file in `folder` into one dict per page.

    Pages are delimited by the `<!-- page: N -->` markers src.convert writes.
    Returns [{"source": "<stem>.pdf", "page": N, "text": str, "images": [...]}]
    in filename then page order. `source` stays the original PDF's name, so
    citations and the eval's gold_source keep pointing at the paper. Figure
    references are lifted out of the text into `images` (paths relative to
    `folder`): the picture itself carries no words to embed.
    """
    pages: list[dict] = []
    for md_path in sorted(Path(folder).glob("*.md")):
        parts = _PAGE_RE.split(md_path.read_text(encoding="utf-8"))
        # split() with one capture group -> [preamble, "1", body1, "2", body2, ...]
        for page_no, body in zip(parts[1::2], parts[2::2]):
            images = _IMAGE_RE.findall(body)
            text = _compact_tables(_COMMENT_RE.sub("", _IMAGE_RE.sub("", body)))
            page = {"source": f"{md_path.stem}.pdf", "page": int(page_no), "text": text.strip()}
            if images:
                page["images"] = images
            pages.append(page)
    return pages


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
    fixed-size word groups. This guarantees every prose unit is bounded, so a
    single run-on sentence can't blow a chunk past the CHUNK_TOKENS * 1.5
    limit the acceptance test checks.
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


def _split_blocks(text: str) -> list[str]:
    """Split page text into blank-line-separated blocks, keeping a `$$ ... $$`
    display formula together even if the LaTeX itself contains a blank line."""
    blocks: list[str] = []
    open_formula: str | None = None
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if open_formula is not None:
            open_formula += "\n\n" + paragraph
            if paragraph.endswith("$$"):
                blocks.append(open_formula)
                open_formula = None
        elif paragraph.startswith("$$") and not (len(paragraph) > 2 and paragraph.endswith("$$")):
            open_formula = paragraph
        else:
            blocks.append(paragraph)
    if open_formula is not None:
        blocks.append(open_formula)
    return blocks


def _is_table(block: str) -> bool:
    """True if every line of `block` is a Markdown pipe-table row."""
    return all(line.lstrip().startswith("|") for line in block.split("\n"))


def _table_pieces(table: str, max_tokens: float) -> list[str]:
    """Split a Markdown table into row groups of ~max_tokens, each repeating
    the header row(s), so every piece still reads as a table with column names."""
    lines = table.split("\n")
    has_rule = len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ")
    header, rows = lines[: 2 if has_rule else 1], lines[2 if has_rule else 1 :]
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and _approx_tokens("\n".join(header + current + [row])) > max_tokens:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(header + current))
    return pieces or [table]


def _page_units(text: str, max_tokens: float) -> list[Unit]:
    """Turn page text into units for packing.

    The separator is "\\n\\n" before a new block and " " between sentences of
    one split paragraph, so a rendered chunk keeps its original layout.
    """
    units: list[Unit] = []
    for block in _split_blocks(text):
        if _is_table(block):
            pieces = [block] if _approx_tokens(block) <= max_tokens else _table_pieces(block, max_tokens)
            units.extend((piece, _ATOMIC, "\n\n") for piece in pieces)
        elif block.startswith("$$"):
            units.append((block, _ATOMIC, "\n\n"))
        elif _HEADING_RE.match(block) and "\n" not in block:
            units.append((block, _HEADING, "\n\n"))
        else:
            for i, piece in enumerate(_units_for_paragraph(block, max_tokens)):
                units.append((piece, _PROSE, "\n\n" if i == 0 else " "))
    return units


def _render(units: list[Unit]) -> str:
    """Join units with their separators back into chunk text."""
    return units[0][0] + "".join(sep + text for text, _, sep in units[1:])


def _overlap_tail(units: list[Unit], overlap_words: int) -> list[Unit]:
    """The last `overlap_words` words of the trailing prose, as one unit to
    start the next chunk with. Tables, formulas and headings are never
    carried over: repeating half a table would be noise, not context."""
    words: list[str] = []
    for text, kind, _ in reversed(units):
        if kind != _PROSE:
            break
        words = text.split() + words
        if len(words) >= overlap_words:
            break
    tail = words[-overlap_words:] if overlap_words else []
    return [(" ".join(tail), _PROSE, "\n\n")] if tail else []


def chunk_pages(
    pages: list[dict], chunk_tokens: int = CHUNK_TOKENS, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Split each page's Markdown into ~chunk_tokens-token chunks with overlap.

    Splits on blocks first (paragraph, table, formula, heading), then sentence
    boundaries (see `_units_for_paragraph`). Chunks never cross a page
    boundary, since a citation needs to point at exactly one page. Each chunk:
    {id: int, text, source, page}, plus `images` (figure files on that page)
    when the page has any.
    """
    chunks: list[dict] = []
    overlap_words = int(chunk_overlap / WORDS_TO_TOKENS)

    for page in pages:
        current: list[Unit] = []
        page_chunks: list[str] = []
        for unit in _page_units(page["text"], chunk_tokens):
            if current and _approx_tokens(_render(current)) + _approx_tokens(unit[0]) > chunk_tokens:
                # A heading ending the chunk would be cut off from its text: move it forward.
                stranded: list[Unit] = []
                while len(current) > 1 and current[-1][1] == _HEADING:
                    stranded.insert(0, current.pop())
                page_chunks.append(_render(current))
                current = stranded or _overlap_tail(current, overlap_words)
            current.append(unit)
        if current:
            page_chunks.append(_render(current))

        for text in page_chunks:
            chunk = {"id": len(chunks), "text": text, "source": page["source"], "page": page["page"]}
            if page.get("images"):
                chunk["images"] = page["images"]
            chunks.append(chunk)

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
    """CLI entry point: convert new PDFs to Markdown, then chunk every .md file
    into parents.json + chunks.json."""
    from src.convert import RAW_DIR, convert_all

    convert_all(RAW_DIR, MD_DIR)
    pages = load_markdown(MD_DIR)
    if not pages:
        print(f"No PDFs found in {RAW_DIR}/ (nothing in {MD_DIR}/). Add PDF files there and re-run.")
        return

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
