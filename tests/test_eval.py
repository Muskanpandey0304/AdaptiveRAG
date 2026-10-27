"""Tests for the pure metric/formatting logic in src/eval.py.

run_config, tune_thresholds, and main() all call the live pipeline
(ask()), which needs a real indexed corpus and, for the LLM-judge path, a
running local LLM -- neither exists yet (data/raw/ and eval/questions.json
are both still empty). Those are exercised for real only once the user
supplies them, per the project's rule against inventing evaluation data.
This file only tests the parts that are pure data processing.
"""

import csv

import src.eval as eval_module
from src.eval import TO_GRADE_PATH, _mean, _p95, hit_at_5, render_results_md, summarize_config, write_to_grade_csv
from src.generator import Answer
from src.pipeline import Result
from src.retrievers import Hit


def _result(hits: list[Hit], confidence: float = 0.5) -> Result:
    return Result(
        answer=Answer(text="some answer", citations=[], tokens_in=10, tokens_out=5, latency_s=0.1),
        route="hybrid",
        reason="default",
        escalated=False,
        confidence=confidence,
        centroid_sim=0.6,
        latency_s=0.1,
        tokens_in=10,
        tokens_out=5,
        hits=hits,
    )


def _hit(source: str, page: int) -> Hit:
    return Hit(chunk_id=0, score=1.0, chunk={"id": 0, "text": "x", "source": source, "page": page})


def test_hit_at_5_true_when_gold_chunk_present() -> None:
    question = {"type": "exact", "gold_source": "doc.pdf", "gold_pages": [3, 4]}
    assert hit_at_5(_result([_hit("doc.pdf", 4)]), question) is True


def test_hit_at_5_false_when_gold_chunk_absent() -> None:
    question = {"type": "exact", "gold_source": "doc.pdf", "gold_pages": [3, 4]}
    assert hit_at_5(_result([_hit("other.pdf", 1)]), question) is False


def test_hit_at_5_skipped_for_general_questions() -> None:
    question = {"type": "general", "gold_source": None, "gold_pages": None}
    assert hit_at_5(_result([_hit("doc.pdf", 4)]), question) is None


def test_mean_and_p95_basic() -> None:
    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert _mean([]) == 0.0
    assert _p95([1, 2, 3, 4, 5]) == 5


def test_summarize_config_aggregates_hit_rate_latency_and_routes() -> None:
    records = [
        {"type": "exact", "latency_s": 0.1, "tokens_in": 10, "tokens_out": 5, "hit_at_5": True, "route": "hybrid", "escalated": False},
        {"type": "exact", "latency_s": 0.3, "tokens_in": 20, "tokens_out": 10, "hit_at_5": False, "route": "hybrid", "escalated": True},
        {"type": "general", "latency_s": 0.2, "tokens_in": 0, "tokens_out": 0, "hit_at_5": None, "route": "direct", "escalated": False},
    ]
    summary = summarize_config(records)
    assert summary["n_questions"] == 3
    assert summary["hit_at_5_overall"] == 0.5
    assert summary["hit_at_5_by_type"] == {"exact": 0.5}
    assert summary["escalation_rate"] == 1 / 3
    assert summary["route_distribution"] == {"hybrid": 2, "direct": 1}


def test_render_results_md_mentions_hand_grading_file_when_not_llm_judged() -> None:
    summaries = {
        "direct": summarize_config(
            [{"type": "general", "latency_s": 0.1, "tokens_in": 5, "tokens_out": 2, "hit_at_5": None, "route": "direct", "escalated": False}]
        )
    }
    report = render_results_md(summaries, llm_judged=None)
    assert "to_grade.csv" in report
    assert "hand-grade" in report.lower()


def test_render_results_md_labels_llm_judged_scores() -> None:
    summaries = {
        "direct": summarize_config(
            [{"type": "general", "latency_s": 0.1, "tokens_in": 5, "tokens_out": 2, "hit_at_5": None, "route": "direct", "escalated": False}]
        )
    }
    report = render_results_md(summaries, llm_judged={"direct": 0.8})
    assert "LLM-judged" in report
    assert "0.80" in report


def test_write_to_grade_csv(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "to_grade.csv"
    monkeypatch.setattr(eval_module, "TO_GRADE_PATH", csv_path)

    records = [
        {
            "id": 1,
            "type": "exact",
            "question": "Q1?",
            "config": "hybrid",
            "answer": "A1",
            "route": "hybrid",
            "reason": "default",
            "escalated": False,
            "confidence": 0.6,
            "latency_s": 0.1,
            "tokens_in": 10,
            "tokens_out": 5,
            "hit_at_5": True,
        }
    ]
    write_to_grade_csv(records)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["question"] == "Q1?"
    assert rows[0]["correct"] == ""


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    from src.eval import bootstrap_ci

    values = [1.0] * 20 + [0.0] * 7
    mean, low, high = bootstrap_ci(values)
    assert mean == 20 / 27
    assert low < mean < high
    assert 0.0 <= low and high <= 1.0
    assert bootstrap_ci(values) == (mean, low, high)  # fixed seed
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)  # no variation -> zero-width
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_mcnemar_exact_matches_known_binomial_values() -> None:
    from src.eval import mcnemar_exact, paired_comparison

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0  # a perfectly even split is no evidence
    assert abs(mcnemar_exact(0, 5) - 0.0625) < 1e-12  # 2 * (1/32)
    assert abs(mcnemar_exact(2, 8) - 0.109375) < 1e-9  # 2 * (1+10+45)/1024
    result = paired_comparison({1: True, 2: True, 3: False}, {1: True, 2: False, 3: True, 4: True})
    assert (result["n"], result["only_a"], result["only_b"]) == (3, 1, 1)
