"""Tests for src/indexer.py.

Built from small synthetic chunks -- data/raw/ is still empty, so there's
no real chunks.json yet to index.
"""

import pickle

import faiss

from src.indexer import build_bm25_index, build_vector_index, tokenize


def _chunks(n: int) -> list[dict]:
    return [
        {"id": i, "text": f"Chunk number {i} talks about widgets and gadgets.", "source": "doc.pdf", "page": 1}
        for i in range(n)
    ]


def test_tokenize_lowercases_and_extracts_word_chars() -> None:
    assert tokenize("Hello, World! v2.0") == ["hello", "world", "v2", "0"]


def test_vector_index_size_matches_chunk_count() -> None:
    chunks = _chunks(6)
    index = build_vector_index(chunks)
    assert index.ntotal == len(chunks)


def test_bm25_index_size_matches_chunk_count() -> None:
    chunks = _chunks(6)
    bm25 = build_bm25_index(chunks)
    assert len(bm25.doc_freqs) == len(chunks)


def test_indexes_round_trip_through_disk(tmp_path) -> None:
    """Saving and reloading each index should preserve its size."""
    chunks = _chunks(4)

    vector_index = build_vector_index(chunks)
    faiss_path = tmp_path / "faiss.index"
    faiss.write_index(vector_index, str(faiss_path))
    loaded_vector_index = faiss.read_index(str(faiss_path))
    assert loaded_vector_index.ntotal == len(chunks)

    bm25_index = build_bm25_index(chunks)
    bm25_path = tmp_path / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(bm25_path, "rb") as f:
        loaded_bm25 = pickle.load(f)
    assert len(loaded_bm25.doc_freqs) == len(chunks)
