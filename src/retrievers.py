"""Vector, BM25, and hybrid (RRF-fused) retrieval over the built indexes."""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import CANDIDATES_K, EMBED_MODEL, QUERY_PREFIX, RERANK_MODEL, RRF_K
from src.indexer import BM25_PATH, CENTROID_PATH, CHUNKS_PATH, FAISS_PATH, tokenize


@dataclass
class Hit:
    """One retrieved chunk plus its score under whichever method found it."""

    chunk_id: int
    score: float
    chunk: dict
    meta: dict = field(default_factory=dict)


def _fuse_rrf(ranked_id_lists: list[list[int]]) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked ID lists.

    Each list contributes 1/(RRF_K + rank + 1) to an ID's score (rank
    counted from 0); an ID missing from a list simply contributes 0 from
    it. Summing means an ID ranked highly in *multiple* lists beats one
    ranked highly in only one, which is the whole point of fusing BM25 and
    vector search instead of picking one.
    """
    scores: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


class Retriever:
    """Loads chunks and both indexes once, then serves all search methods.

    Relies on the invariant from src/ingest.py that chunk ids are exactly
    0..len(chunks)-1 in list order, which is also the order both indexes
    were built in -- so a chunk's id doubles as its position in both.
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        faiss_path: Path = FAISS_PATH,
        bm25_path: Path = BM25_PATH,
        centroid_path: Path = CENTROID_PATH,
    ) -> None:
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)
        self.faiss_index = faiss.read_index(str(faiss_path))
        with open(bm25_path, "rb") as f:
            self.bm25_index = pickle.load(f)

        # Optional: indexes built before this feature existed won't have
        # one yet. centroid_similarity() raises a clear error rather than
        # silently returning a meaningless number if it's missing.
        self.centroid: np.ndarray | None = np.load(centroid_path) if Path(centroid_path).exists() else None

        # Both models are sizeable downloads/loads; only pay for them if a
        # method that actually needs them gets called.
        self._embed_model: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(EMBED_MODEL)
        return self._embed_model

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def vector_search(self, query: str, k: int) -> list[Hit]:
        """Embed `query` with the BGE query prefix and search FAISS.

        FAISS's IndexFlatIP already returns results sorted best-first
        (highest inner product = most similar, since embeddings are
        normalized so inner product equals cosine similarity).
        """
        query_vec = self.embed_model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self.faiss_index.search(query_vec, k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if k exceeds the index size
                continue
            hits.append(Hit(chunk_id=int(idx), score=float(score), chunk=self.chunks[idx]))
        return hits

    def centroid_similarity(self, query: str) -> float:
        """Cosine similarity between `query` and the corpus's overall
        centroid embedding -- "is this topic broadly like my corpus at
        all," independent of whether any single chunk happens to match.

        Raises if the index was built before centroid.npy existed; the
        fix is re-running `python -m src.indexer`, not silently returning
        a meaningless number.
        """
        if self.centroid is None:
            raise RuntimeError(f"{CENTROID_PATH} not found. Re-run `python -m src.indexer` to build it.")
        query_vec = self.embed_model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")[0]
        return float(np.dot(query_vec, self.centroid))

    def bm25_search(self, query: str, k: int) -> list[Hit]:
        """Score every chunk with BM25 and return the top k, best first."""
        scores = self.bm25_index.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:k]
        return [Hit(chunk_id=int(i), score=float(scores[i]), chunk=self.chunks[i]) for i in top_indices]

    def hybrid_search(self, query: str, k: int) -> list[Hit]:
        """BM25 + vector search, fused with RRF.

        Pulls CANDIDATES_K from each (wider than k, so fusion has room to
        reorder) and keeps each hit's original vector cosine in
        hit.meta["vector_sim"] (0.0 if the chunk never showed up in the
        vector candidates) for the pipeline's confidence check later.
        """
        vector_hits = self.vector_search(query, CANDIDATES_K)
        bm25_hits = self.bm25_search(query, CANDIDATES_K)

        vector_sim_by_id = {hit.chunk_id: hit.score for hit in vector_hits}
        chunk_by_id = {hit.chunk_id: hit.chunk for hit in vector_hits + bm25_hits}

        rrf_scores = _fuse_rrf([[h.chunk_id for h in vector_hits], [h.chunk_id for h in bm25_hits]])
        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        return [
            Hit(
                chunk_id=chunk_id,
                score=rrf_scores[chunk_id],
                chunk=chunk_by_id[chunk_id],
                meta={"vector_sim": vector_sim_by_id.get(chunk_id, 0.0)},
            )
            for chunk_id in ranked_ids
        ]

    def expand_neighbors(self, hits: list[Hit], window: int = 1) -> list[Hit]:
        """Add each hit's immediately adjacent chunks (same source, within
        `window` positions on either side), so an answer split across a
        chunk boundary isn't missing half its context.

        Chunk ids are sequential within a source document (see
        src/ingest.py), so id-1/id+1 are physically adjacent chunks -- but
        only when they share the *same* source, since the last chunk of
        one PDF and the first chunk of the next happen to sit at adjacent
        ids too.

        A neighbor inherits its anchor hit's score (it was never scored by
        retrieval itself) and is tagged via hit.meta["neighbor_of"] so
        callers can tell an anchor from an added neighbor. Already-included
        ids are never duplicated. Hits are returned grouped by cluster in
        ascending chunk_id order (not retrieval-rank order), so the
        context reads like a contiguous passage.
        """
        seen_ids = {hit.chunk_id for hit in hits}
        expanded: list[Hit] = []
        for hit in hits:
            cluster_ids = [hit.chunk_id]
            for offset in range(1, window + 1):
                for neighbor_id in (hit.chunk_id - offset, hit.chunk_id + offset):
                    if not (0 <= neighbor_id < len(self.chunks)):
                        continue
                    if self.chunks[neighbor_id]["source"] != hit.chunk["source"]:
                        continue
                    cluster_ids.append(neighbor_id)

            for neighbor_id in sorted(cluster_ids):
                if neighbor_id in seen_ids and neighbor_id != hit.chunk_id:
                    continue  # already added via an earlier cluster or another hit
                seen_ids.add(neighbor_id)
                if neighbor_id == hit.chunk_id:
                    expanded.append(hit)
                else:
                    expanded.append(
                        Hit(
                            chunk_id=neighbor_id,
                            score=hit.score,
                            chunk=self.chunks[neighbor_id],
                            meta={"neighbor_of": hit.chunk_id},
                        )
                    )
        return expanded

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        """Re-score `hits` with a cross-encoder and return the top k.

        The cross-encoder model is loaded lazily (only the first time
        rerank() is actually called) since most queries never use it.
        """
        if not hits:
            return []
        pairs = [(query, hit.chunk["text"]) for hit in hits]
        scores = self.reranker.predict(pairs)
        order = np.argsort(scores)[::-1][:k]
        return [
            Hit(chunk_id=hits[i].chunk_id, score=float(scores[i]), chunk=hits[i].chunk, meta=hits[i].meta)
            for i in order
        ]
