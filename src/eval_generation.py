"""Compare answer-generation settings on identical retrieved context.

`src.eval` runs the whole pipeline, so a change in the answer can come from
retrieval or from generation. This script fixes retrieval: it retrieves once
per question set, caches the parent hits, and then runs every variant (prompt
version, context mode, grounding check) on exactly the same context. Only the
generator differs between variants.

Exact questions are scored automatically against the regex answer keys in
eval/answer_patterns.json (every pattern must appear in the answer).
Conceptual questions have no key and are left for hand grading.

Usage:
    python -m src.eval_generation dev A_v1r_full B_v3_full
    python -m src.eval_generation heldout A_v1r_full C_v3_sent
To compare models, pass --model (e.g. --model llama3.1:8b); each run takes its
settings as an argument, so nothing global is mutated.
"""

import argparse
import dataclasses
import json
import pickle
import re
import statistics
import time
from pathlib import Path

from src import config, pipeline
from src.config import RERANK_POOL_K, TOP_K

QUESTION_FILES = {"dev": Path("eval/questions.json"), "heldout": Path("eval/heldout_questions.json")}
PATTERNS_PATH = Path("eval/answer_patterns.json")
OUT_DIR = Path("eval/generation")

# Variant name -> config overrides applied before generating. These are the
# settings compared in the "Getting more out of the 3B model" README section.
VARIANTS: dict[str, dict] = {
    "A_v1r_full": dict(prompt_version="v1r", context_mode="full", grounding_check=False),
    "B_v3_full": dict(prompt_version="v3", context_mode="full", grounding_check=False),
    "C_v3_sent": dict(prompt_version="v3", context_mode="sentences", grounding_check=False),
    "D_v3_sent_ground": dict(prompt_version="v3", context_mode="sentences", grounding_check=True),
    "E_v1r_sent": dict(prompt_version="v1r", context_mode="sentences", grounding_check=False),
    "F_v3_full_ground": dict(prompt_version="v3", context_mode="full", grounding_check=True),
    "G_v1r_auto": dict(prompt_version="v1r", context_mode="auto", grounding_check=False),
    # G is the current default (v1r + adaptive context); H changes only the
    # citation sentence of the prompt (v1c), so G vs H isolates that sentence.
    "H_v1c_auto": dict(prompt_version="v1c", context_mode="auto", grounding_check=False),
}

NOT_FOUND_ANSWERS = {"NOT_FOUND", pipeline.NOT_FOUND_TEXT}


def load_in_scope(dataset: str) -> list[dict]:
    """The questions that have a gold answer (the off-topic `general` ones don't)."""
    questions = json.loads(QUESTION_FILES[dataset].read_text(encoding="utf-8"))
    return [q for q in questions if q["type"] != "general"]


def cached_hits(dataset: str, questions: list[dict]) -> dict[int, list]:
    """Retrieve once per question (hybrid -> rerank -> parents, as the pipeline
    does) and cache the result, so every variant gets identical context. The
    cache is tied to the index: delete it after re-indexing."""
    cache = OUT_DIR / f"hits_{dataset}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    retriever = pipeline._get_retriever()
    hits_by_id = {}
    for q in questions:
        pool = retriever.hybrid_search(q["question"], RERANK_POOL_K)
        pool = retriever.rerank(q["question"], pool, len(pool))
        hits_by_id[q["id"]] = retriever.to_parents(pool)[:TOP_K]
    with open(cache, "wb") as f:
        pickle.dump(hits_by_id, f)
    return hits_by_id


def variant_settings(name: str, model: str | None = None) -> config.Settings:
    """The defaults with this variant's overrides (and optionally another model) applied."""
    base = config.Settings.from_config()
    overrides = dict(VARIANTS[name], **({"llm_model": model} if model else {}))
    return dataclasses.replace(base, **overrides)


def run_variant(
    name: str, questions: list[dict], hits_by_id: dict[int, list], model: str | None = None
) -> dict[int, dict]:
    """Generate an answer for every question under one variant's settings."""
    settings = variant_settings(name, model)
    results = {}
    for q in questions:
        answer = pipeline._generate_answer(q["question"], hits_by_id[q["id"]], settings)
        results[q["id"]] = {
            "text": answer.text,
            "n_citations": len(answer.citations),
            "tokens_in": answer.tokens_in,
            "tokens_out": answer.tokens_out,
            "latency_s": answer.latency_s,
        }
    return results


def exact_correct(text: str, patterns: list[str]) -> bool:
    """True if every answer-key pattern appears in the answer."""
    flat = text.replace("\n", " ")
    return all(re.search(p, flat) for p in patterns)


def summarize(results: dict[int, dict], patterns: dict[str, list[str]]) -> dict:
    """Automatic metrics for one variant: exact-answer score, refusals, literal
    "[n]" citations (the prompt's placeholder copied verbatim), latency, tokens."""
    keyed = [(qid, r) for qid, r in results.items() if str(qid) in patterns]
    texts = [r["text"] for r in results.values()]
    return {
        "exact_correct": f"{sum(exact_correct(r['text'], patterns[str(qid)]) for qid, r in keyed)}/{len(keyed)}",
        "refusals": sum(t.strip() in NOT_FOUND_ANSWERS for t in texts),
        "literal_n_citations": sum(bool(re.search(r"\[n\d*\]|\[n=\d+\]", t)) for t in texts),
        "latency_mean_s": round(statistics.mean(r["latency_s"] for r in results.values()), 1),
        "tokens_in_mean": round(statistics.mean(r["tokens_in"] for r in results.values())),
        "tokens_out_mean": round(statistics.mean(r["tokens_out"] for r in results.values())),
    }


def main() -> None:
    """CLI entry point: run each named variant on one question set."""
    parser = argparse.ArgumentParser(description="Compare generation variants on cached retrieval.")
    parser.add_argument("dataset", choices=list(QUESTION_FILES))
    parser.add_argument("variants", nargs="+", choices=list(VARIANTS))
    parser.add_argument("--model", default=None, help="Ollama model for this run (default: LLM_MODEL)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_in_scope(args.dataset)
    hits_by_id = cached_hits(args.dataset, questions)
    patterns = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    print(f"{args.dataset}: {len(questions)} in-scope questions, model {args.model or config.LLM_MODEL}", flush=True)

    for name in args.variants:
        start = time.time()
        results = run_variant(name, questions, hits_by_id, args.model)
        out_path = OUT_DIR / f"{args.dataset}_{(args.model or config.LLM_MODEL).replace(':', '-')}_{name}.json"
        out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{name} ({time.time() - start:.0f}s): {summarize(results, patterns)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
