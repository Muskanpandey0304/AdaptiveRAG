"""Convert PDFs to layout-aware Markdown (the corpus the rest of the pipeline reads).

Run as a script (`python -m src.convert`) to turn every PDF in data/raw/ into
data/md/<name>.md. It is incremental: a PDF whose sha256 matches the manifest
is skipped, so adding one new paper only converts that paper.

Why Markdown, and what each element becomes:
  * text / headings   -> plain Markdown, reading order fixed by layout analysis
                         (two-column papers no longer interleave columns);
  * tables            -> pipe tables (rows and columns survive, unlike pypdf);
  * formulas          -> LaTeX between $ ... $ / $$ ... $$ (GPU only, see
                         _formulas_enabled);
  * figures           -> PNG files under data/md/<name>_assets/, referenced as
                         ![...](...) so a UI can show them;
  * scanned pages     -> OCR (docling runs OCR on image-only regions).

Every page starts with an `<!-- page: N -->` marker. Chunks must cite exactly
one page, and the marker is how `src.ingest` recovers the page number.
"""

import hashlib
import json
import os
from pathlib import Path

RAW_DIR = Path("data/raw")
MD_DIR = Path("data/md")
MANIFEST_PATH = MD_DIR / "manifest.json"

# Formula -> LaTeX uses a vision-language model that took ~195 s PER FORMULA on
# a CPU (an 8-page paper: 17 minutes, ~10 of them on 3 formulas), so it is on
# only when a GPU is present. Force with CONVERT_FORMULAS=1 / 0. Without it a
# formula is left out of the Markdown (docling's "formula-not-decoded"
# placeholder, stripped at ingest); the sentences around it are unaffected.
def _formulas_enabled() -> bool:
    setting = os.getenv("CONVERT_FORMULAS", "auto").lower()
    if setting == "auto":
        import torch

        return torch.cuda.is_available()
    return setting in ("1", "true", "yes")


# Bump when the conversion settings below change, so every PDF is re-converted.
# The formula mode is part of it: switching it re-converts the corpus.
CONVERTER_VERSION = "docling-2" + ("+formulas" if _formulas_enabled() else "")

PAGE_MARKER = "<!-- page: {n} -->"


def file_sha256(path: Path) -> str:
    """Content hash of `path`, used to detect PDFs that changed or are new."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Read the manifest ({pdf name: {sha256, pages, converter}}), {} if absent."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def needs_conversion(pdf_path: Path, manifest: dict, md_dir: Path = MD_DIR) -> bool:
    """True if `pdf_path` is new, changed, converted with older settings, or
    its .md file has gone missing."""
    entry = manifest.get(pdf_path.name)
    if entry is None or not (md_dir / f"{pdf_path.stem}.md").exists():
        return True
    return entry["sha256"] != file_sha256(pdf_path) or entry["converter"] != CONVERTER_VERSION


def _build_converter():
    """Docling converter with table structure, formula->LaTeX, OCR and figure
    export on. Imported lazily: it is a heavy import and the tests never need it."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=_formulas_enabled(),
        generate_picture_images=True,
        images_scale=2.0,
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def convert_pdf(converter, pdf_path: Path, md_dir: Path = MD_DIR) -> int:
    """Convert one PDF to md_dir/<stem>.md (+ figure PNGs); return its page count."""
    from docling_core.types.doc import ImageRefMode

    document = converter.convert(str(pdf_path)).document
    assets_dir = md_dir / f"{pdf_path.stem}_assets"
    page_numbers = sorted(document.pages)

    sections = []
    for page_no in page_numbers:
        body = document.export_to_markdown(
            page_no=page_no,
            image_mode=ImageRefMode.REFERENCED,
            # Raw text, not display-safe text: "max\_tokens" and "&lt;" would
            # hurt BM25 and embeddings, and nothing here renders HTML.
            escape_underscores=False,
            escape_html=False,
            image_dir=assets_dir,
            image_uri_prefix=f"{assets_dir.name}/",  # trailing "/": docling concatenates, it doesn't join
        )
        sections.append(f"{PAGE_MARKER.format(n=page_no)}\n\n{body.strip()}\n")

    md_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / f"{pdf_path.stem}.md").write_text("\n".join(sections), encoding="utf-8")
    return len(page_numbers)


def convert_all(raw_dir: Path = RAW_DIR, md_dir: Path = MD_DIR, force: bool = False) -> list[str]:
    """Convert every new/changed PDF in `raw_dir`; return the names converted."""
    pdfs = sorted(raw_dir.glob("*.pdf"))
    manifest = load_manifest(md_dir / "manifest.json")
    todo = [p for p in pdfs if force or needs_conversion(p, manifest, md_dir)]
    if not todo:
        return []

    converter = _build_converter()
    converted = []
    for i, pdf_path in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] converting {pdf_path.name} ...", flush=True)
        pages = convert_pdf(converter, pdf_path, md_dir)
        manifest[pdf_path.name] = {
            "sha256": file_sha256(pdf_path),
            "pages": pages,
            "converter": CONVERTER_VERSION,
        }
        # Saved after every paper so an interrupted run resumes where it stopped.
        (md_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        converted.append(pdf_path.name)
    return converted


def main() -> None:
    """CLI entry point: python -m src.convert [--force]."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs in data/raw/ to Markdown in data/md/.")
    parser.add_argument("--force", action="store_true", help="Re-convert every PDF, even unchanged ones.")
    args = parser.parse_args()

    if not list(RAW_DIR.glob("*.pdf")):
        print(f"No PDFs found in {RAW_DIR}/. Add PDF files there and re-run.")
        return
    converted = convert_all(force=args.force)
    print(f"Converted {len(converted)} PDF(s)." if converted else "All PDFs already converted; nothing to do.")


if __name__ == "__main__":
    main()
