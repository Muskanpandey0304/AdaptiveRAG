# PROJECT SPEC: AdaptiveRAG (query-routed RAG from scratch)

## 0. Role and rules for the agent

You are building a learning project for a student's resume. The student must be able to explain every line in an interview.

**Hard rules**
1. Do NOT use LangChain, LlamaIndex, Haystack, or any RAG framework. Use only: `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `pypdf`, `numpy`, `streamlit`, `pytest`, `python-dotenv`, and one LLM SDK.
2. Work **one step at a time**. After each step: run its acceptance tests, print the results, commit to git, then **stop and wait for the user to say "next"**.
3. Never invent data, questions, or evaluation numbers. If `data/raw/` or `eval/questions.json` is missing or empty when needed, stop and ask the user.
4. Keep code small and readable: type hints, a docstring on every function, comments explaining *why*. No clever one-liners.
5. No hardcoded API keys. Load from `.env`.
6. If a design choice is not specified here, pick the simplest option and note it in `DECISIONS.md` (one line: choice and reason).
7. Pin versions in `requirements.txt` after installing.

**Project inputs**
- DOMAIN of documents: `[e.g. "FastAPI documentation PDFs"]`
- LLM provider: `[anthropic | openai | gemini | ollama]`

## 1. Target architecture

```
query
  → router.route(query) → (route, reason)
      ├─ "direct"  → llm answers with no context
      ├─ "keyword" → bm25_search → confidence check ─ low → escalate to hybrid
      └─ "hybrid"  → bm25 + vector → RRF fusion → (optional rerank)
                       → confidence check ─ low → "not found" answer
  → generator builds prompt from chunks → LLM → answer + citations
  → log to logs/queries.jsonl
```

## 2. Folder structure

```
adaptive-rag/
├── data/raw/                # user's PDFs (user provides)
├── data/index/              # generated: chunks.json, faiss.index, bm25.pkl
├── src/
│   ├── config.py
│   ├── llm.py
│   ├── ingest.py
│   ├── indexer.py
│   ├── retrievers.py
│   ├── generator.py
│   ├── router.py
│   ├── pipeline.py
│   └── eval.py
├── tests/
├── eval/questions.json      # user provides
├── eval/results.md          # generated
├── logs/queries.jsonl       # generated
├── app.py
├── DECISIONS.md
├── requirements.txt
├── .env.example
└── README.md
```

## 3. Step-by-step build

### STEP 1: Setup and config

Create the structure, venv instructions, `requirements.txt`, `.env.example` (`LLM_PROVIDER`, `LLM_MODEL`, `API_KEY`), and `src/config.py` with these constants:

```python
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-base"
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 50
TOP_K = 5
CANDIDATES_K = 20          # retrieved before fusion/rerank
RRF_K = 60
BM25_MIN_SCORE = 5.0       # placeholder, tuned in Step 8
VECTOR_MIN_SIM = 0.45      # placeholder, tuned in Step 8
```

**Accept:** `python -c "import src.config"` works.

### STEP 2: LLM wrapper (`src/llm.py`)

One function: `generate(prompt: str, system: str | None = None) -> LLMResult`
`LLMResult = dataclass(text: str, input_tokens: int, output_tokens: int, latency_s: float)`
Select the provider from `LLM_PROVIDER`. Implement only the chosen provider. Retry once on transient errors.

**Accept:** `tests/test_llm.py` calls `generate("Say OK")` and asserts non-empty text (skipped if no key).

### STEP 3: Ingest and chunk (`src/ingest.py`)

- `load_pdfs(folder) -> list[dict]`: one dict per page: `{source, page, text}`. Source is the filename; pages are 1-indexed.
- `clean_text(text) -> str`: collapse whitespace, fix hyphenated line breaks, drop lines that repeat on more than 50% of pages (headers/footers).
- `chunk_pages(pages) -> list[dict]`: split on paragraph, then sentence boundaries, to about `CHUNK_TOKENS` tokens (approximate tokens as words × 1.3) with `CHUNK_OVERLAP`. Never cross page boundaries. Each chunk: `{id: int, text, source, page}`.
- CLI: `python -m src.ingest` writes `data/index/chunks.json`.

**Accept:** `tests/test_ingest.py`: no empty chunks; every chunk has all four fields; ids unique and sequential; no chunk exceeds `CHUNK_TOKENS * 1.5`. Print 5 random chunks for the user to eyeball.

### STEP 4: Indexing (`src/indexer.py`)

- `build_vector_index(chunks)`: embed chunk texts (no prefix) with `normalize_embeddings=True`, build `faiss.IndexFlatIP`, save to `data/index/faiss.index`.
- `build_bm25_index(chunks)`: tokenize with lowercase and regex `\w+`, build `BM25Okapi`, pickle to `data/index/bm25.pkl`.
- Put the tokenizer in one shared function and use it for both indexing and querying.
- CLI: `python -m src.indexer`.

**Accept:** a test loads both indexes and checks `index.ntotal == len(chunks)`.

### STEP 5: Retrievers (`src/retrievers.py`)

Class `Retriever` loads chunks and both indexes once. Every search method returns `list[Hit]`, where `Hit = dataclass(chunk_id, score, chunk)`, sorted best first:
- `vector_search(query, k)`: embed with `QUERY_PREFIX`, search FAISS, score = cosine.
- `bm25_search(query, k)`: score = raw BM25.
- `hybrid_search(query, k)`: take `CANDIDATES_K` from each, fuse with Reciprocal Rank Fusion (`1/(RRF_K + rank + 1)`, ranks from 0), return top k. Also keep each hit's original vector cosine in `hit.meta["vector_sim"]` (0 if absent) for the confidence check.
- `rerank(query, hits, k)`: CrossEncoder on `(query, chunk.text)`, return top k. Load the model lazily.

**Accept:** `tests/test_retrievers.py` checks that results are sorted, k is respected, there are no duplicate ids in hybrid, and RRF gives a higher fused score to an item ranked 1st in both lists than to one ranked 1st in only one. Then print, for 5 user-provided queries, the top-3 of each method side by side.

### STEP 6: Generator (`src/generator.py`)

`answer_with_context(query, hits) -> Answer`
`Answer = dataclass(text, citations: list[dict], tokens_in, tokens_out, latency_s)`
- Prompt: numbered context blocks `[1] ...`, then rules: answer only from context; cite as `[n]`; if the context lacks the answer, reply exactly `NOT_FOUND`.
- Parse `[n]` citations and map them to `{n, source, page}`. Ignore citation numbers that don't exist.
- `answer_direct(query) -> Answer`: no context, no citations.

**Accept:** a mocked-LLM test checks citation parsing and `NOT_FOUND` handling. Then a live baseline demo: run plain vector top-5 RAG on 3 user questions. **Tag this commit `baseline-simple-rag`.**

### STEP 7: Router and pipeline (`src/router.py`, `src/pipeline.py`)

`route(query) -> tuple[str, str]` returns `(route, reason)`. Rule order (first match wins):
1. **direct**: greeting or thanks (regex); pure arithmetic; "who are you / what can you do". Reason string states which rule fired.
2. **keyword**: query contains a quoted phrase; or matches an identifier pattern (`[A-Z]{1,6}[-_]?\d{2,}`, version like `v?\d+\.\d+(\.\d+)?`, snake_case or CamelCase token, file path, or a CLI flag like `--word`); or has 3 words or fewer and no question word (what/why/how/when/who/which/explain).
3. **hybrid**: everything else.

All patterns are named constants at the top of the file, and `DECISIONS.md` explains the reason for each.

`pipeline.ask(query, force_route: str | None = None, use_rerank: bool = False) -> Result`
`Result = dataclass(answer, route, reason, escalated: bool, confidence: float | None, latency_s, tokens_in, tokens_out, hits)`

Flow:
- **direct** → `answer_direct`.
- **keyword** → `bm25_search`; if top BM25 score < `BM25_MIN_SCORE`, set `escalated=True` and run the hybrid flow.
- **hybrid** → `hybrid_search` (+ rerank if enabled); confidence = top `vector_sim` (or the reranker score if reranked); if confidence < `VECTOR_MIN_SIM`, skip the LLM and return the answer "Not found in the documents." Also return that if the LLM replies `NOT_FOUND`.
- Append one JSON line per call to `logs/queries.jsonl`: query, route, reason, escalated, confidence, latency, tokens, top chunk ids.
- `force_route` lets the eval run each strategy in isolation ("direct", "keyword", "hybrid", "vector"); with it, the router and the escalation are bypassed.

**Accept:** `tests/test_router.py` with at least 25 labeled queries (written by the agent, reviewed by the user) must pass, plus a pipeline test that checks an out-of-domain question returns "Not found" and not an invented answer.

### STEP 8: Evaluation (`src/eval.py`)

**Prerequisite:** the user provides `eval/questions.json` of 40+ items:
```json
{"id": 1, "question": "...", "type": "general|exact|conceptual",
 "expected_answer": "...", "gold_source": "file.pdf", "gold_pages": [12, 13]}
```
For `general` questions, `gold_source` is null. If the file is missing, **stop and ask**.

Run these configs over all questions: `direct`, `bm25`, `vector`, `hybrid`, `hybrid+rerank`, `router`.

Metrics per config and per question type:
- **hit@5**: a retrieved chunk matches `gold_source` and one of `gold_pages` (skip for `general`).
- **correctness**: by default, export all answers to `eval/to_grade.csv` with an empty `correct` column for the **user to grade by hand** (0/1). A separate flag `--llm-judge` may score with a fixed rubric, but results must be labeled "LLM-judged" in the output.
- **latency**: mean and p95.
- **tokens**: mean in and out.
- **router only**: route distribution and escalation rate.

Output `eval/results.md` (a table per metric) and print it. Then add a `tune_thresholds()` helper that sweeps `BM25_MIN_SCORE` and `VECTOR_MIN_SIM` and reports hit rate vs. false-answer rate on out-of-scope questions. **Do not edit `config.py` automatically.** Print the suggestion and let the user decide.

**Accept:** the script runs end to end on the user's real questions, and `results.md` is generated from real runs only.

### STEP 9: UI (`app.py`)

Streamlit: a query box, the answer, a citations list (source + page), and a side panel showing route, reason, escalated, confidence, latency, and tokens. A checkbox for rerank. An expander showing the retrieved chunks.

**Accept:** `streamlit run app.py` works; an out-of-scope question shows "Not found".

### STEP 10: README and polish

README sections: what it is, architecture diagram (ASCII or mermaid), setup, how to run each CLI step, evaluation table (copied from `results.md`), "what worked / what didn't" (written from real results only), limitations, and **Related work** (adaptive-rag-engine on PyPI, ArgusLens, and Augur, and what this project does differently: a no-retrieval route and confidence-based fallback). Make sure `DECISIONS.md` is complete and the repo has a `.gitignore` (`.env`, `data/index/`, `venv/`).

## 4. Definition of done
- All tests pass.
- The simple-RAG baseline is tagged and evaluated against the router.
- `results.md` contains real numbers from the user's questions.
- No framework dependencies; `requirements.txt` is pinned.
- The student can explain chunking, BGE query prefix, normalized embeddings with inner product, RRF, router rules, and how thresholds were chosen.

## 5. Final instruction to the agent
Start with STEP 1 only. When it is done, report what you created, show the test output, and wait.
