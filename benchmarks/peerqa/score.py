"""Score the PeerQA answers: answerability (does it refuse the unanswerable questions?) and answer overlap.

Three things, all automatic, none human- or judge-graded:
  * Answerability: a refusal ("Not found in the documents.") counts as the prediction "unanswerable". Reported as
    precision/recall/F1 of the unanswerable class, plus how often answerable questions were wrongly refused.
  * Answer quality: Rouge-L F1 against the author's answer, taking the better of the author's own free-form answer
    and the GPT-4o-augmented version PeerQA ships. A refused answerable question scores 0. Rouge-L rewards word
    overlap, so a correct answer worded differently scores low: read it as a rough lower bound.
  * Retrieval: copied from retrieval_results.json (evidence sentences found in the top 5 passages).

    PYTHONPATH=. python benchmarks/peerqa/score.py
"""

import json
import re
import statistics
from pathlib import Path

from src.eval import bootstrap_ci

HERE = Path(__file__).parent
NOT_FOUND = "not found in the documents."


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(prediction: str, reference: str) -> float:
    """Rouge-L F1: longest common subsequence of word tokens."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for p in pred:
        current = [0]
        for j, r in enumerate(ref, 1):
            current.append(previous[j - 1] + 1 if p == r else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_refusal(text: str) -> bool:
    return re.sub(r"\[\d+\]", "", text).strip().lower() == NOT_FOUND


def main() -> None:
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    answers = json.loads((HERE / "answers.json").read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        a = answers.get(q["question_id"])
        if a is None:
            continue
        refused = is_refusal(a["text"])
        references = [r for r in (q["gold_free_form"], q["gold_augmented"]) if r]
        score = 0.0 if refused else max((rouge_l(a["text"], r) for r in references), default=0.0)
        rows.append({"answerable": q["answerable"], "refused": refused, "rouge": score, "venue": q["venue"],
                     "latency_s": a["latency_s"], "gated": a["gated"]})

    def mean_ci(values: list[float]) -> str:
        if not values:
            return "n/a"
        mean, low, high = bootstrap_ci(values)
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    true_pos = sum(r["refused"] for r in unanswerable)  # unanswerable and refused
    false_pos = sum(r["refused"] for r in answerable)  # answerable but refused
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / len(unanswerable) if unanswerable else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    answered = [r for r in answerable if not r["refused"]]
    summary = {
        "model": next(iter(answers.values()))["model"],
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "unanswerable_refused": true_pos,
        "answerable_wrongly_refused": false_pos,
        "answerability_precision": precision,
        "answerability_recall": recall,
        "answerability_f1": f1,
        "answerability_accuracy": (true_pos + len(answerable) - false_pos) / len(rows),
        "rouge_l_all_answerable": statistics.mean(r["rouge"] for r in answerable),
        "rouge_l_when_answered": statistics.mean(r["rouge"] for r in answered) if answered else None,
        "mean_latency_s_when_generated": statistics.mean(r["latency_s"] for r in rows if not r["gated"]),
    }
    retrieval = json.loads((HERE / "retrieval_results.json").read_text(encoding="utf-8"))

    def retrieval_line(label: str, key: str) -> str:
        return f"- {label}: {mean_ci(retrieval[key]['per_question'])}"

    lines = [
        "# PeerQA results",
        "",
        f"{summary['n_questions']} questions on 90 papers ({summary['n_answerable']} answerable, {summary['n_unanswerable']} unanswerable);",
        f"each question searches only its own paper. Generator: {summary['model']}. 95% bootstrap intervals over questions.",
        "No human or LLM judging: Rouge-L is word overlap with the author's answer (see score.py).",
        "",
        f"## Retrieval ({retrieval['n_with_evidence']} answerable questions with mapped evidence)",
        "",
        retrieval_line("At least one gold evidence sentence in the top 5 passages, hybrid", "hybrid_any"),
        retrieval_line("At least one gold evidence sentence in the top 5 passages, hybrid + rerank", "rerank_any"),
        retrieval_line("Share of gold evidence sentences in the top 5 passages, hybrid + rerank", "rerank_recall"),
        "",
        "## Answers",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Unanswerable questions correctly refused | {true_pos} of {len(unanswerable)} ({mean_ci([float(r['refused']) for r in unanswerable])}) |",
        f"| Answerable questions wrongly refused | {false_pos} of {len(answerable)} ({mean_ci([float(r['refused']) for r in answerable])}) |",
        f"| Answerability F1 (unanswerable class), precision, recall | {f1:.3f}, {precision:.3f}, {recall:.3f} |",
        f"| Rouge-L, all answerable (refusal = 0) | {mean_ci([r['rouge'] for r in answerable])} |",
        f"| Rouge-L, answerable questions the system answered (n={len(answered)}) | {mean_ci([r['rouge'] for r in answered])} |",
        "",
        f"Mean latency when the model generated: {summary['mean_latency_s_when_generated']:.1f} s.",
    ]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
