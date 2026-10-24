"""Run evaluation configs over eval/questions.json and report metrics.

This only produces meaningful output once the user supplies a real
eval/questions.json (40+ items) and real PDFs in data/raw/ -- per the
project's rule against inventing evaluation data, the CLI refuses to run
without a real questions file rather than fabricating one.
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.config import BM25_MIN_SCORE, VECTOR_MIN_SIM
from src.pipeline import Result, ask

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_PATH = Path("eval/results.md")
TO_GRADE_PATH = Path("eval/to_grade.csv")

# Config name -> ask() kwargs. "vector" and "bm25" reuse pipeline.py's
# force_route options directly; "router" lets the router decide (no
# force_route), matching how the app actually calls ask() day to day.
CONFIGS: dict[str, dict] = {
    "direct": {"force_route": "direct", "use_rerank": False},
    "bm25": {"force_route": "keyword", "use_rerank": False},
    "vector": {"force_route": "vector", "use_rerank": False},
    "hybrid": {"force_route": "hybrid", "use_rerank": False},
    "hybrid+rerank": {"force_route": "hybrid", "use_rerank": True},
    "router": {"force_route": None, "use_rerank": False},
}


def load_questions() -> list[dict]:
    """Load eval/questions.json; refuse loudly if it's missing or empty
    rather than inventing questions."""
    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"{QUESTIONS_PATH} not found. Add 40+ real questions there before running eval.")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    if not questions:
        raise SystemExit(f"{QUESTIONS_PATH} is empty. Add real questions before running eval.")
    return questions


def hit_at_5(result: Result, question: dict) -> bool | None:
    """Whether any retrieved chunk matches the question's gold source/page.

    Returns None (skip) for `general` questions, which by design have no
    single right source/page to check against.
    """
    if question["type"] == "general" or question.get("gold_source") is None:
        return None
    gold_source = question["gold_source"]
    gold_pages = set(question.get("gold_pages") or [])
    return any(hit.chunk["source"] == gold_source and hit.chunk["page"] in gold_pages for hit in result.hits)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p95(values: list[float]) -> float:
    """95th percentile via nearest-rank. Good enough at eval-sized (tens
    to low hundreds of questions) samples; no need for interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return ordered[index]


def run_config(config_name: str, questions: list[dict]) -> list[dict]:
    """Run every question through one config; one record per question."""
    kwargs = CONFIGS[config_name]
    records = []
    for question in questions:
        result = ask(question["question"], **kwargs)
        records.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "config": config_name,
                "answer": result.answer.text,
                "route": result.route,
                "reason": result.reason,
                "escalated": result.escalated,
                "confidence": result.confidence,
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "hit_at_5": hit_at_5(result, question),
            }
        )
    return records


def summarize_config(records: list[dict]) -> dict:
    """Aggregate one config's per-question records into result.md's
    metrics: hit@5 (overall and per question type), latency, tokens, and
    route/escalation stats (meaningful mainly for the "router" config)."""
    latencies = [r["latency_s"] for r in records]
    tokens_in = [r["tokens_in"] for r in records]
    tokens_out = [r["tokens_out"] for r in records]

    checked = [r for r in records if r["hit_at_5"] is not None]
    hit_at_5_overall = _mean([1.0 if r["hit_at_5"] else 0.0 for r in checked])

    hit_at_5_by_type = {}
    for q_type in sorted({r["type"] for r in checked}):
        type_records = [r for r in checked if r["type"] == q_type]
        hit_at_5_by_type[q_type] = _mean([1.0 if r["hit_at_5"] else 0.0 for r in type_records])

    return {
        "n_questions": len(records),
        "hit_at_5_overall": hit_at_5_overall,
        "hit_at_5_by_type": hit_at_5_by_type,
        "latency_mean_s": _mean(latencies),
        "latency_p95_s": _p95(latencies),
        "tokens_in_mean": _mean(tokens_in),
        "tokens_out_mean": _mean(tokens_out),
        "route_distribution": dict(Counter(r["route"] for r in records)),
        "escalation_rate": _mean([1.0 if r["escalated"] else 0.0 for r in records]),
    }


def write_to_grade_csv(all_records: list[dict]) -> None:
    """Export every (config, question) answer with an empty `correct`
    column for the user to hand-grade (0/1) -- the default scoring path,
    since correctness numbers are never invented automatically."""
    TO_GRADE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "type", "question", "config", "answer", "correct"]
    with open(TO_GRADE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_records:
            writer.writerow({**{k: record[k] for k in fieldnames if k != "correct"}, "correct": ""})


_JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a candidate answer correctly answers a question, "
    "given the expected answer. Reply with exactly one character: 1 if the "
    "candidate answer is correct, 0 if it is not."
)


def llm_judge_correctness(all_records: list[dict], questions_by_id: dict[int, dict]) -> dict[str, float]:
    """Score each config's answers against expected_answer with a fixed
    rubric via the LLM. Kept separate from hand-graded correctness --
    callers must label this "LLM-judged" wherever it's reported."""
    from src.llm import generate  # local import: only needed on this path

    scores_by_config: dict[str, list[float]] = defaultdict(list)
    for record in all_records:
        question = questions_by_id[record["id"]]
        prompt = (
            f"Question: {question['question']}\n"
            f"Expected answer: {question['expected_answer']}\n"
            f"Candidate answer: {record['answer']}"
        )
        verdict = generate(prompt, system=_JUDGE_SYSTEM_PROMPT).text.strip()
        scores_by_config[record["config"]].append(1.0 if verdict.startswith("1") else 0.0)
    return {config: _mean(scores) for config, scores in scores_by_config.items()}


def render_results_md(summaries: dict[str, dict], llm_judged: dict[str, float] | None) -> str:
    """Build eval/results.md: one table per metric, across all configs."""
    lines = ["# Evaluation Results", ""]

    lines += ["## hit@5 (overall)", "", "| Config | hit@5 |", "|---|---|"]
    for config, summary in summaries.items():
        lines.append(f"| {config} | {summary['hit_at_5_overall']:.2f} |")
    lines.append("")

    all_types = sorted({t for s in summaries.values() for t in s["hit_at_5_by_type"]})
    if all_types:
        lines += ["## hit@5 by question type", "", "| Config | " + " | ".join(all_types) + " |", "|---|" + "---|" * len(all_types)]
        for config, summary in summaries.items():
            row = [f"{summary['hit_at_5_by_type'].get(t, 0.0):.2f}" for t in all_types]
            lines.append(f"| {config} | " + " | ".join(row) + " |")
        lines.append("")

    lines += ["## Latency (seconds)", "", "| Config | mean | p95 |", "|---|---|---|"]
    for config, summary in summaries.items():
        lines.append(f"| {config} | {summary['latency_mean_s']:.3f} | {summary['latency_p95_s']:.3f} |")
    lines.append("")

    lines += ["## Tokens", "", "| Config | mean in | mean out |", "|---|---|---|"]
    for config, summary in summaries.items():
        lines.append(f"| {config} | {summary['tokens_in_mean']:.1f} | {summary['tokens_out_mean']:.1f} |")
    lines.append("")

    if "router" in summaries:
        router_summary = summaries["router"]
        lines += ["## Router route distribution", "", "| Route | Count |", "|---|---|"]
        for r, count in sorted(router_summary["route_distribution"].items()):
            lines.append(f"| {r} | {count} |")
        lines.append("")
        lines.append(f"Escalation rate: {router_summary['escalation_rate']:.2%}")
        lines.append("")

    if llm_judged is not None:
        lines += [
            "## Correctness (LLM-judged)",
            "",
            "_Scored by a fixed-rubric LLM judge, not hand-graded._",
            "",
            "| Config | correct |",
            "|---|---|",
        ]
        for config, score in llm_judged.items():
            lines.append(f"| {config} | {score:.2f} |")
        lines.append("")
    else:
        lines.append(f"Correctness: hand-grade `{TO_GRADE_PATH}` (fill in the empty `correct` column with 0/1).")
        lines.append("")

    return "\n".join(lines)


def tune_thresholds(questions: list[dict]) -> str:
    """Sweep BM25_MIN_SCORE and VECTOR_MIN_SIM and report, for each
    candidate threshold, the answer rate on in-scope ("scoped") questions
    vs. the false-answer rate on out-of-scope ("general") questions --
    the tradeoff a higher threshold makes. Confidence is computed once per
    question (it doesn't depend on the threshold); only the gate decision
    does, so candidates are compared cheaply. Never edits config.py --
    prints a suggestion for the user to apply by hand.
    """
    general_questions = [q for q in questions if q["type"] == "general"]
    scoped_questions = [q for q in questions if q["type"] != "general"]

    bm25_scoped = [ask(q["question"], force_route="keyword").confidence for q in scoped_questions]
    bm25_general = [ask(q["question"], force_route="keyword").confidence for q in general_questions]
    vector_scoped = [ask(q["question"], force_route="hybrid").confidence for q in scoped_questions]
    vector_general = [ask(q["question"], force_route="hybrid").confidence for q in general_questions]

    def _answer_rate(confidences: list[float | None], threshold: float) -> float:
        return _mean([1.0 if c is not None and c >= threshold else 0.0 for c in confidences])

    lines = ["## Threshold sweep suggestions (not applied automatically)", ""]
    lines.append(f"Current: BM25_MIN_SCORE={BM25_MIN_SCORE}, VECTOR_MIN_SIM={VECTOR_MIN_SIM}")
    lines.append("")
    lines.append("### BM25_MIN_SCORE (keyword route)")
    lines.append("")
    lines.append("| threshold | answer rate (scoped) | false-answer rate (general) |")
    lines.append("|---|---|---|")
    for factor in (0.5, 0.75, 1.0, 1.25, 1.5):
        threshold = BM25_MIN_SCORE * factor
        lines.append(f"| {threshold:.2f} | {_answer_rate(bm25_scoped, threshold):.2%} | {_answer_rate(bm25_general, threshold):.2%} |")
    lines.append("")

    lines.append("### VECTOR_MIN_SIM (hybrid route)")
    lines.append("")
    lines.append("| threshold | answer rate (scoped) | false-answer rate (general) |")
    lines.append("|---|---|---|")
    for factor in (0.5, 0.75, 1.0, 1.1, 1.25):
        threshold = VECTOR_MIN_SIM * factor
        lines.append(f"| {threshold:.3f} | {_answer_rate(vector_scoped, threshold):.2%} | {_answer_rate(vector_general, threshold):.2%} |")
    lines.append("")
    lines.append("Pick thresholds manually based on this table; config.py is not edited automatically.")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point: run every config over eval/questions.json."""
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveRAG across retrieval configs.")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Also score correctness with a fixed-rubric LLM judge (labeled 'LLM-judged' in results.md).",
    )
    args = parser.parse_args()

    questions = load_questions()
    questions_by_id = {q["id"]: q for q in questions}

    summaries = {}
    all_records: list[dict] = []
    for config_name in CONFIGS:
        print(f"Running config: {config_name} ({len(questions)} questions)...")
        records = run_config(config_name, questions)
        all_records.extend(records)
        summaries[config_name] = summarize_config(records)

    write_to_grade_csv(all_records)
    llm_judged = llm_judge_correctness(all_records, questions_by_id) if args.llm_judge else None

    report = render_results_md(summaries, llm_judged)
    report += "\n\n" + tune_thresholds(questions) + "\n"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nWrote {RESULTS_PATH} and {TO_GRADE_PATH}")


if __name__ == "__main__":
    main()
