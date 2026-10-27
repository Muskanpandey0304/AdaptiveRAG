"""Retrieval-only evaluation: no LLM, so it runs in minutes.

Reports, for the dev and held-out question sets:
  * hit@5 of hybrid search and of hybrid + rerank (overall and per question
    type), each with a 95% bootstrap interval;
  * the score distributions the confidence thresholds are chosen from
    (top rerank probability, best-child cosine, centroid cosine), split into
    in-scope vs off-topic questions.

Use it to judge any change to chunking, titles, embeddings, BM25 or the
reranker before spending 20+ minutes on an end-to-end eval, and to choose
thresholds (VECTOR_MIN_SIM, RERANK_MIN_PROB, CENTROID_MIN_SIM) from data.

    python -m src.retrieval_eval            # writes eval/results_retrieval.md
"""

import json
from pathlib import Path

import numpy as np

from src.config import RERANK_POOL_K, TOP_K
from src.eval import bootstrap_ci
from src.pipeline import _get_retriever
from src.retrievers import Hit

QUESTION_FILES = {"dev": Path("eval/questions.json"), "held-out": Path("eval/heldout_questions.json")}
OUT_PATH = Path("eval/results_retrieval.md")


def gold_hit(hits: list[Hit], question: dict) -> bool:
    """Whether any hit is on the question's gold source and one of its gold pages."""
    pages = set(question["gold_pages"])
    return any(hit.chunk["source"] == question["gold_source"] and hit.chunk["page"] in pages for hit in hits)


def describe(values: list[float]) -> str:
    """min / p10 / median / p90 / max of a list of scores, for threshold choice."""
    if not values:
        return "n/a"
    low, p10, median, p90, high = np.percentile(values, [0, 10, 50, 90, 100])
    return f"min {low:.3f} / p10 {p10:.3f} / median {median:.3f} / p90 {p90:.3f} / max {high:.3f}"


def evaluate(questions: list[dict]) -> dict:
    """Run every question through retrieval once and collect hits and scores."""
    retriever = _get_retriever()
    outcomes = {"hybrid": [], "hybrid+rerank": []}
    by_type: dict[str, dict[str, list[float]]] = {}
    scores = {"in_scope": {"rerank": [], "cosine": [], "centroid": []}, "off_topic": {"rerank": [], "cosine": [], "centroid": []}}

    for question in questions:
        text = question["question"]
        pool = retriever.hybrid_search(text, RERANK_POOL_K)
        reranked = retriever.rerank(text, pool, len(pool)) if pool else []
        group = "off_topic" if question["type"] == "general" else "in_scope"

        scores[group]["rerank"].append(reranked[0].score if reranked else 0.0)
        top_parents = retriever.to_parents(pool)[:TOP_K]
        scores[group]["cosine"].append(top_parents[0].meta.get("vector_sim", 0.0) if top_parents else 0.0)
        scores[group]["centroid"].append(retriever.centroid_similarity(text))

        if group == "in_scope":
            plain = 1.0 if gold_hit(top_parents, question) else 0.0
            ranked = 1.0 if gold_hit(retriever.to_parents(reranked)[:TOP_K], question) else 0.0
            outcomes["hybrid"].append(plain)
            outcomes["hybrid+rerank"].append(ranked)
            slot = by_type.setdefault(question["type"], {"hybrid": [], "hybrid+rerank": []})
            slot["hybrid"].append(plain)
            slot["hybrid+rerank"].append(ranked)

    return {"outcomes": outcomes, "by_type": by_type, "scores": scores}


def render(name: str, result: dict) -> list[str]:
    """Markdown section for one question set."""
    n = len(result["outcomes"]["hybrid"])
    lines = [f"## {name} ({n} in-scope questions)", "", "| Method | hit@5 | 95% CI |", "|---|---|---|"]
    for method, values in result["outcomes"].items():
        mean, low, high = bootstrap_ci(values)
        lines.append(f"| {method} | {mean:.3f} | [{low:.3f}, {high:.3f}] |")
    lines += ["", "| Type | hybrid | hybrid+rerank |", "|---|---|---|"]
    for q_type, methods in sorted(result["by_type"].items()):
        k = len(methods["hybrid"])
        lines.append(
            f"| {q_type} ({k}) | {sum(methods['hybrid']):.0f}/{k} | {sum(methods['hybrid+rerank']):.0f}/{k} |"
        )
    lines += ["", "Score distributions (choose thresholds between the two groups):", ""]
    for metric in ("rerank", "cosine", "centroid"):
        for group in ("in_scope", "off_topic"):
            lines.append(f"- {metric}, {group}: {describe(result['scores'][group][metric])}")
    lines.append("")
    return lines


def main() -> None:
    """CLI entry point: evaluate retrieval on both question sets."""
    lines = ["# Retrieval-only evaluation", ""]
    for name, path in QUESTION_FILES.items():
        questions = json.loads(path.read_text(encoding="utf-8"))
        lines += render(name, evaluate(questions))
    report = "\n".join(lines)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
