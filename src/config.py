"""Central configuration for AdaptiveRAG.

Two kinds of settings live here:
1. Secrets/environment choices (provider, model, key) loaded from `.env`,
   so nothing is hardcoded in source.
2. Fixed knobs for chunking, retrieval, and fusion, so every module tunes
   against the same numbers instead of each hardcoding its own.
"""

import os

from dotenv import load_dotenv

# Load variables from a `.env` file in the project root into the process
# environment. If `.env` does not exist (e.g. fresh checkout), this is a
# no-op and os.environ is left untouched.
load_dotenv()

# --- Secrets / provider selection (never hardcode these) -------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1")
API_KEY: str = os.getenv("API_KEY", "")

# --- Embedding and reranking models ----------------------------------------

# BGE small: a strong, small, open embedding model. "small" keeps the project
# runnable on a laptop CPU without a GPU.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models were trained asymmetrically: queries need this instruction
# prefix prepended so the query embedding lands in the same space as the
# (unprefixed) passage embeddings. Passages are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Cross-encoder reranker, used only when `use_rerank=True` in the pipeline.
RERANK_MODEL = "BAAI/bge-reranker-base"

# --- Chunking ----------------------------------------------------------------

# Target chunk size in (approximate) tokens, and overlap between consecutive
# chunks so an answer near a chunk boundary isn't split away from its context.
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 50

# --- Retrieval ---------------------------------------------------------------

# Number of chunks returned to the generator after fusion/rerank.
TOP_K = 5

# Number of candidates pulled from each retriever (BM25, vector) before
# fusion narrows them down to TOP_K. Wider than TOP_K so fusion has room to
# reorder.
CANDIDATES_K = 20

# Reciprocal Rank Fusion constant: higher RRF_K flattens the score curve so
# rank differences matter less; 60 is the commonly used default from the
# original RRF paper.
RRF_K = 60

# --- Confidence thresholds (placeholders; tuned for real in Step 8) --------

BM25_MIN_SCORE = 5.0
VECTOR_MIN_SIM = 0.45

# Corpus-level signal (added post-spec): minimum cosine similarity between
# a query and the whole corpus's centroid embedding (see
# src.indexer.compute_centroid) before an answer is attempted at all, on
# top of VECTOR_MIN_SIM's per-chunk check. Catches "no chunk broadly
# matches this corpus's topic" even when one stray chunk's wording happens
# to overlap a query lexically/semantically.
#
# Measured against the real 55-question eval set: general (out-of-scope)
# questions scored 0.44-0.59, scoped (real) questions scored 0.59-0.78 --
# a clean gap. 0.55 sits comfortably above the actual problem cases
# (factual/creative off-topic questions, 0.44-0.51) and below every scoped
# question (min 0.59), without hugging the single borderline case ("Thanks,
# that was helpful!", 0.589 -- already caught by the router's greeting
# regex before this gate is ever reached in normal operation). See
# DECISIONS.md.
CENTROID_MIN_SIM = 0.55
