# Design Decisions

One line per decision not already pinned down by the spec: the choice and the reason.

- **LLM provider: Ollama.** User chose it in setup so the project needs no API key and runs fully offline/local.
- **Python version: 3.10 (via venv), not the system default 3.14.** 3.14 is too new; `faiss-cpu` and `torch` (a `sentence-transformers` dependency) don't yet publish wheels for it.
- **LLM SDK for Ollama: the `ollama` PyPI package.** It's the official lightweight client and keeps `src/llm.py` a thin wrapper rather than hand-rolled HTTP calls.
- **`.gitignore` scoped to secrets and generated artifacts** (`.env`, `venv/`, `data/index/`, `logs/*.jsonl`, `eval/results.md`, `eval/to_grade.csv`) from Step 1 onward, not deferred to Step 10, so nothing generated or secret gets committed while building step by step.
