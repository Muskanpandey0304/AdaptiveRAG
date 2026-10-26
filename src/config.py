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

# --- Generation limits --------------------------------------------------------

# Hard cap on generated tokens per call. Without one, a small local model that
# falls into a repetition loop keeps going until its context fills: one
# held-out eval answer ran to ~209,000 characters (and dragged that config's
# mean latency to 63s). Real answers here are well under 200 tokens.
LLM_MAX_TOKENS = 512

# Answer-prompt version for the generator ("v1", "v1r" or "v2"; see
# DECISIONS.md for the A/B). "v1" is the original prompt. "v1r" is v1 plus one
# retry when the reply has no answer text (a 3B model often answers with just
# "[1]"). "v2" is a reworded prompt (complete sentences, no citation-only
# replies) plus the same retry. Override with the PROMPT_VERSION env var so
# eval can compare versions on identical code.
PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1r")

# "v3" = structured JSON output ({found, answer, sources}); it removes
# bare/literal-"[n]" citations and makes a refusal an explicit found=false.
# Whether it (and the options below) becomes the default is decided by the
# measurements recorded in DECISIONS.md.

# What the generator reads. "full" sends the whole parent chunks; "sentences"
# sends only the SENTENCE_TOP_N best-matching sentences from them (each with
# SENTENCE_WINDOW neighbors either side); "auto" uses "sentences" for
# single-fact lookup questions (router.is_factoid_question) and "full" for
# everything else. Measured: "sentences" matches or beats "full" on lookups and
# is ~4x faster, but loses ~2 of 10 conceptual answers, hence "auto".
CONTEXT_MODE: str = os.getenv("CONTEXT_MODE", "auto")
SENTENCE_TOP_N = 6
SENTENCE_WINDOW = 1

# After generating, check every number in the answer appears in the retrieved
# parents; if not, retry once. Free (no extra model), catches wrong numbers.
GROUNDING_CHECK: bool = os.getenv("GROUNDING_CHECK", "0") == "1"

# Deterministic decoding, so eval numbers don't move between runs and a
# repeated question gets a repeated answer.
LLM_TEMPERATURE = 0.0

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
#
# Parent-child chunking: the *child* chunks (CHILD_TOKENS) are what gets
# embedded and BM25-indexed -- small spans match a query precisely. Each
# child points at a larger *parent* chunk (CHUNK_TOKENS), and the parent's
# text is what the LLM actually reads, so an answer is never cut off at a
# small chunk's edge. CHUNK_TOKENS/CHUNK_OVERLAP size the parents.
CHUNK_TOKENS = 600
CHUNK_OVERLAP = 60
CHILD_TOKENS = 120
CHILD_OVERLAP = 20

# --- Retrieval ---------------------------------------------------------------

# Number of chunks returned to the generator after fusion/rerank.
TOP_K = 5

# Number of candidates pulled from each retriever (BM25, vector) before
# fusion narrows them down to TOP_K. Wider than TOP_K so fusion has room to
# reorder.
CANDIDATES_K = 20

# How many fused (child) candidates are handed to the reranker. The
# cross-encoder only helps if it sees more than the TOP_K it will keep --
# reranking exactly TOP_K hits just shuffles the same five.
RERANK_POOL_K = 30

# Reciprocal Rank Fusion constant: higher RRF_K flattens the score curve so
# rank differences matter less; 60 is the commonly used default from the
# original RRF paper.
RRF_K = 60

# --- Confidence thresholds (placeholders; tuned for real in Step 8) --------

# Re-measured on the parent-child index (55 eval questions, retrieval only;
# see DECISIONS.md). In-sample, so keep a margin and re-check if the corpus
# or chunk sizes change.
#
# BM25 top-parent score: scoped questions min 17.7, so 5.0 never fired. At 15
# only a weak keyword match escalates to the hybrid path (which has the
# reranker + centroid gates), and no scoped question did in this sample.
BM25_MIN_SCORE = 15.0
# Best-child cosine: scoped min 0.665 (p10 0.73); 0.60 answers every scoped
# question and drops half of the off-topic ones that 0.45 let through.
VECTOR_MIN_SIM = 0.60

# Confidence floor on the rerank path (bge-reranker relevance probability,
# 0-1). Scoped questions' top hit scored >= 0.84; off-topic ones were
# 0.01/0.04/0.47 or >= 0.95 (those are caught by the centroid gate below),
# so anything in 0.5-0.8 is equivalent on this data; 0.5 is the model's
# natural boundary.
RERANK_MIN_PROB = 0.5

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
