"""Build the FAISS vector index and BM25 index from data/index/chunks.json.

Run as a script (`python -m src.indexer`) after `python -m src.ingest`.
"""

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import DOC_PREFIX, EMBED_MODEL
from src.ingest import contextualize

CHUNKS_PATH = Path("data/index/chunks.json")
FAISS_PATH = Path("data/index/faiss.index")
BM25_PATH = Path("data/index/bm25.pkl")
CENTROID_PATH = Path("data/index/centroid.npy")
META_PATH = Path("data/index/meta.json")  # which embedding model built the index

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase + \\w+ tokenizer.

    Shared by indexing and querying (src/retrievers.py imports this) so
    BM25 sees the exact same vocabulary rules at search time as at build
    time -- a mismatch here would silently break retrieval.
    """
    return _TOKEN_RE.findall(text.lower())


def build_vector_index(chunks: list[dict]) -> faiss.Index:
    """Embed chunk texts and build a FAISS inner-product index over them.

    The query-side prefix is never used here -- these are passages (BGE's
    asymmetric prefix only applies to queries); DOC_PREFIX is the model's
    passage template, empty for BGE and "passage: " for E5. Embeddings are
    L2-normalized, which makes inner product equivalent to cosine
    similarity, so a plain IndexFlatIP can be used instead of a slower
    similarity computation.
    """
    model = SentenceTransformer(EMBED_MODEL)
    texts = [DOC_PREFIX + contextualize(chunk) for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Build a BM25 index over tokenized chunk texts."""
    tokenized_chunks = [tokenize(contextualize(chunk)) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def compute_centroid(vector_index: faiss.Index) -> np.ndarray:
    """Average embedding across every chunk in `vector_index`, re-normalized
    to unit length.

    This is a corpus-level "what is this collection broadly about" signal,
    distinct from any single chunk's similarity to a query. A query whose
    embedding is far from this centroid is topically unlike the corpus as a
    whole, even if one stray chunk happens to score high against it (e.g. a
    benchmark paper's example Q&A pair that lexically resembles an unrelated
    question). Reconstructed directly from the index rather than
    re-embedding, since IndexFlatIP already stores the normalized vectors.
    """
    embeddings = vector_index.reconstruct_n(0, vector_index.ntotal)
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return centroid.astype("float32")


def main() -> None:
    """CLI entry point: build and save both indexes from chunks.json."""
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} not found. Run `python -m src.ingest` first.")
        return

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        print(f"{CHUNKS_PATH} is empty. Add PDFs to data/raw/ and re-run ingest.")
        return

    vector_index = build_vector_index(chunks)
    faiss.write_index(vector_index, str(FAISS_PATH))

    centroid = compute_centroid(vector_index)
    np.save(CENTROID_PATH, centroid)

    bm25_index = build_bm25_index(chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)

    META_PATH.write_text(json.dumps({"embed_model": EMBED_MODEL, "n_chunks": len(chunks)}), encoding="utf-8")

    print(f"Indexed {len(chunks)} chunks -> {FAISS_PATH}, {BM25_PATH}, {CENTROID_PATH}")


if __name__ == "__main__":
    main()
