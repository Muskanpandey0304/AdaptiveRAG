"""Tests for src/eval_generation.py's scoring (generation itself needs Ollama)."""

import json
from pathlib import Path

from src.eval_generation import exact_correct, summarize


def test_exact_correct_needs_every_pattern() -> None:
    assert exact_correct("HELM evaluates 30 models on 42 scenarios.", [r"\b30\b", r"\b42\b"])
    assert not exact_correct("HELM evaluates 30 models.", [r"\b30\b", r"\b42\b"])
    assert exact_correct("between 2 and\n8 steps", [r"2 (and|to) 8"])  # newlines flattened


def test_summarize_counts_correct_refusals_and_literal_citations() -> None:
    results = {
        1: {"text": "It has 57 subjects [1].", "n_citations": 1, "tokens_in": 10, "tokens_out": 5, "latency_s": 1.0},
        2: {"text": "Not found in the documents.", "n_citations": 0, "tokens_in": 10, "tokens_out": 5, "latency_s": 3.0},
        3: {"text": "[n] Something else.", "n_citations": 0, "tokens_in": 10, "tokens_out": 5, "latency_s": 2.0},
    }
    summary = summarize(results, {"1": [r"\b57\b"], "2": [r"\b80\b"]})
    assert summary["exact_correct"] == "1/2"
    assert summary["refusals"] == 1
    assert summary["literal_n_citations"] == 1
    assert summary["latency_mean_s"] == 2.0


def test_every_answer_pattern_is_a_valid_regex_for_a_real_question() -> None:
    import re

    patterns = json.loads(Path("eval/answer_patterns.json").read_text(encoding="utf-8"))
    ids = {q["id"] for f in ("eval/questions.json", "eval/heldout_questions.json")
           for q in json.loads(Path(f).read_text(encoding="utf-8"))}
    for qid, pats in patterns.items():
        assert int(qid) in ids
        for p in pats:
            re.compile(p)


def test_variant_settings_override_defaults_and_model() -> None:
    from src.eval_generation import VARIANTS, variant_settings

    settings = variant_settings("B_v3_full", model="llama3.1:8b")
    assert settings.prompt_version == "v3" and settings.context_mode == "full"
    assert settings.llm_model == "llama3.1:8b"
    assert all(set(v) <= {"prompt_version", "context_mode", "grounding_check"} for v in VARIANTS.values())
