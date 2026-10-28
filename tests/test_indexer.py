"""Tests for src/indexer.py.

Built from small synthetic chunks -- data/raw/ is still empty, so there's
no real chunks.json yet to index.
"""


import faiss
import numpy as np

from src.indexer import build_bm25_index, build_vector_index, compute_centroid, tokenize


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


def test_compute_centroid_is_unit_length_and_matches_embedding_dim() -> None:
    chunks = _chunks(6)
    index = build_vector_index(chunks)
    centroid = compute_centroid(index)
    assert centroid.shape == (index.d,)
    assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-5)


def test_compute_centroid_closer_to_on_topic_than_off_topic_query() -> None:
    """A sanity check that the centroid behaves like a topic signal: a
    query about the corpus's actual subject should be more similar to the
    centroid than a wildly unrelated one."""
    chunks = [
        {"id": i, "text": text, "source": "doc.pdf", "page": 1}
        for i, text in enumerate(
            [
                "FastAPI is a modern Python web framework for building APIs.",
                "Uvicorn is the ASGI server used to run FastAPI applications.",
                "Pydantic models validate FastAPI request and response bodies.",
                "Dependency injection in FastAPI is handled by the Depends function.",
            ]
        )
    ]
    index = build_vector_index(chunks)
    centroid = compute_centroid(index)

    from sentence_transformers import SentenceTransformer

    from src.config import EMBED_MODEL, QUERY_PREFIX

    model = SentenceTransformer(EMBED_MODEL)
    on_topic = model.encode([QUERY_PREFIX + "How do I validate a request body in FastAPI?"], normalize_embeddings=True)[0]
    off_topic = model.encode([QUERY_PREFIX + "What is the boiling point of mercury on Mars?"], normalize_embeddings=True)[0]

    assert np.dot(on_topic, centroid) > np.dot(off_topic, centroid)


def test_indexes_round_trip_through_disk(tmp_path) -> None:
    """Saving and reloading each index should preserve its size."""
    chunks = _chunks(4)

    vector_index = build_vector_index(chunks)
    faiss_path = tmp_path / "faiss.index"
    faiss.write_index(vector_index, str(faiss_path))
    loaded_vector_index = faiss.read_index(str(faiss_path))
    assert loaded_vector_index.ntotal == len(chunks)

    assert len(build_bm25_index(chunks).doc_freqs) == len(chunks)

    centroid = compute_centroid(vector_index)
    centroid_path = tmp_path / "centroid.npy"
    np.save(centroid_path, centroid)
    loaded_centroid = np.load(centroid_path)
    assert np.allclose(loaded_centroid, centroid)
