# AdaptiveRAG

A query-routed RAG pipeline built from scratch (no LangChain/LlamaIndex/Haystack).

> This README is a work in progress — full architecture, usage, and evaluation
> sections are added in the final build step.

## Setup

```bash
# Use Python 3.10 (faiss-cpu / torch wheels aren't available for newer versions yet)
py -3.10 -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env: set LLM_PROVIDER, LLM_MODEL, API_KEY (Ollama needs no key)
```

See [DECISIONS.md](DECISIONS.md) for the reasoning behind choices not pinned
down by the project spec.
