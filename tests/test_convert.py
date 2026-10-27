"""Tests for src/convert.py's incremental logic (the docling conversion itself
is too heavy to run in unit tests; it is exercised by `python -m src.convert`)."""

import json

from src.convert import CONVERTER_VERSION, file_sha256, load_manifest, needs_conversion


def _setup(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    return pdf, md_dir


def test_new_pdf_needs_conversion(tmp_path) -> None:
    pdf, md_dir = _setup(tmp_path)
    assert needs_conversion(pdf, {}, md_dir)


def test_unchanged_pdf_is_skipped(tmp_path) -> None:
    pdf, md_dir = _setup(tmp_path)
    (md_dir / "a.md").write_text("x", encoding="utf-8")
    manifest = {"a.pdf": {"sha256": file_sha256(pdf), "pages": 1, "converter": CONVERTER_VERSION}}
    assert not needs_conversion(pdf, manifest, md_dir)


def test_changed_pdf_or_old_converter_or_missing_md_is_reconverted(tmp_path) -> None:
    pdf, md_dir = _setup(tmp_path)
    md = md_dir / "a.md"
    md.write_text("x", encoding="utf-8")
    good = {"sha256": file_sha256(pdf), "pages": 1, "converter": CONVERTER_VERSION}

    assert needs_conversion(pdf, {"a.pdf": {**good, "sha256": "stale"}}, md_dir)
    assert needs_conversion(pdf, {"a.pdf": {**good, "converter": "docling-0"}}, md_dir)
    md.unlink()
    assert needs_conversion(pdf, {"a.pdf": good}, md_dir)


def test_load_manifest_missing_and_present(tmp_path) -> None:
    assert load_manifest(tmp_path / "nope.json") == {}
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"a.pdf": {}}), encoding="utf-8")
    assert load_manifest(path) == {"a.pdf": {}}
