# Design Decisions

One line per decision not already pinned down by the spec: the choice and the reason.

- **LLM provider: Ollama.** User chose it in setup so the project needs no API key and runs fully offline/local.
- **Python version: 3.10 (via venv), not the system default 3.14.** 3.14 is too new; `faiss-cpu` and `torch` (a `sentence-transformers` dependency) don't yet publish wheels for it.
- **LLM SDK for Ollama: the `ollama` PyPI package.** It's the official lightweight client and keeps `src/llm.py` a thin wrapper rather than hand-rolled HTTP calls.
- **`.gitignore` scoped to secrets and generated artifacts** (`.env`, `venv/`, `data/index/`, `logs/*.jsonl`, `eval/results.md`, `eval/to_grade.csv`) from Step 1 onward, not deferred to Step 10, so nothing generated or secret gets committed while building step by step.
- **`test_llm.py` skip condition: "no local Ollama server", not "no API key".** The spec's accept criterion was written for key-based providers; Ollama has no key, so the thing that actually blocks the test on a fresh machine is the server not running (checked via `ollama.list()`). Same intent (skip, don't fail, when the provider isn't usable here), adapted to the provider that was actually chosen.
- **Retry-once implementation: a plain `try` then one unconditional retry on any `Exception`,** not a narrower exception type. The `ollama` client doesn't expose a distinct "transient vs. permanent" error type, so narrowing would be guesswork; a single retry is cheap even for a non-transient error (fails the same way twice, no harm done).
- **Latency measured with `time.perf_counter()` around the call**, not Ollama's own `total_duration` field. Wall-clock time is what the rest of the pipeline (and the eval step) cares about, and it works uniformly if a future provider doesn't report its own timing.
