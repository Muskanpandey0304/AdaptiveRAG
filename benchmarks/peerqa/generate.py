"""Phase 2 of the PeerQA benchmark: answer every question with the 8B model on the cached per-paper context.

Reads cache/hits.pkl (written by retrieve.py), applies the pipeline's rerank-confidence gate (RERANK_MIN_PROB),
and otherwise generates with the project's real answer path (default prompt and adaptive context) using
llama3.1:8b. The corpus-calibrated gates (best-child cosine, centroid) are not applied: their thresholds were
tuned on our own 26 papers and mean nothing for a different corpus. Resumable: answers already in answers.json
are kept. Run with the main environment (it needs Ollama):

    PYTHONPATH=. venv/Scripts/python.exe benchmarks/peerqa/generate.py [--model llama3.1:8b]
"""

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

from benchmarks.peerqa.retrieve import open_paper_retriever
from src import pipeline
from src.config import RERANK_MIN_PROB, Settings

HERE = Path(__file__).parent
ANSWERS = HERE / "answers.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()

    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    cached = pickle.load(open(HERE / "cache" / "hits.pkl", "rb"))
    settings = dataclasses.replace(Settings.from_config(), llm_model=args.model)
    answers = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}
    shared_reranker = None  # sentence selection loads the cross-encoder once, not once per paper

    for n, q in enumerate(questions, 1):
        if q["question_id"] in answers:
            continue
        entry = cached[q["question_id"]]
        if entry["top_rerank"] < RERANK_MIN_PROB:
            record = {"text": "Not found in the documents.", "gated": True, "latency_s": 0.0}
        else:
            retriever = open_paper_retriever(q["paper_id"])  # sentence selection must use this paper's chunks
            retriever._reranker = shared_reranker
            pipeline._retriever = retriever
            start = time.perf_counter()
            result = pipeline._generate_answer(q["question"], entry["hits"], settings)
            record = {"text": result.text, "gated": False, "latency_s": round(time.perf_counter() - start, 1)}
            shared_reranker = retriever._reranker or shared_reranker
        record["model"] = args.model
        answers[q["question_id"]] = record
        ANSWERS.write_text(json.dumps(answers, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{n}/{len(questions)} {record['latency_s']}s gated={record['gated']}", flush=True)


if __name__ == "__main__":
    main()
