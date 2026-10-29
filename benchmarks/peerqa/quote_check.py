"""Experiment: quote-then-answer as the answerability decision on PeerQA.

The rerank-score gate cannot tell answerable from unanswerable PeerQA questions (AUC 0.47: inside one paper every
passage is on topic). Here the model must first copy, word for word, the sentences that answer the question; code
then checks each quote really occurs in the retrieved passages. No verified quote -> abstain. So the decision rests
on text we can check, not on a relevance score. The rerank gate is not applied (this check replaces it).

Run on a balanced subset (all unanswerable questions plus the same number of answerable ones, every third by id) to
keep it near an hour; resumable, so the rest can be added later. Uses the cached retrieval from retrieve.py.

    PYTHONPATH=. venv/Scripts/python.exe benchmarks/peerqa/quote_check.py [--model llama3.1:8b]
"""

import argparse
import json
import pickle
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path

from benchmarks.peerqa.score import rouge_l
from src.eval import bootstrap_ci
from src.llm import generate

HERE = Path(__file__).parent
OUT = HERE / "quote_answers.json"
NOT_FOUND = "Not found in the documents."
MIN_QUOTE_WORDS = 5

SYSTEM = "You answer questions about one research paper using only the passages you are given."
SCHEMA = {
    "type": "object",
    "properties": {
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"passage": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["passage", "text"],
            },
        },
        "answer": {"type": "string"},
    },
    "required": ["quotes", "answer"],
}


def build_prompt(question: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return (
        f"Question: {question}\n\nPassages:\n{numbered}\n\n"
        "Step 1: copy, word for word, up to 3 sentences from the passages that directly answer the question. "
        "Copy them exactly; do not paraphrase or combine sentences. If no passage contains information that "
        "answers the question, return an empty quotes list.\n"
        "Step 2: answer the question in 1-3 sentences using only those quotes. If the quotes list is empty, the "
        f'answer must be exactly "{NOT_FOUND}"\nReturn JSON.'
    )


def norm(text: str) -> str:
    """Lowercase, punctuation to spaces, collapse whitespace (the extracted text has 'words .' style spacing)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def quote_verified(quote: str, passages_norm: str) -> bool:
    """True if the quote (at least MIN_QUOTE_WORDS words) appears in the passages, allowing ~10% small differences."""
    q = norm(quote)
    if len(q.split()) < MIN_QUOTE_WORDS:
        return False
    if q in passages_norm:
        return True
    match = SequenceMatcher(None, q, passages_norm, autojunk=False).find_longest_match(0, len(q), 0, len(passages_norm))
    return match.size >= 0.9 * len(q)


def pick_subset(questions: list[dict]) -> list[dict]:
    """All unanswerable questions plus as many answerable ones (every third by id), interleaved."""
    unanswerable = sorted((q for q in questions if not q["answerable"]), key=lambda q: q["question_id"])
    answerable = sorted((q for q in questions if q["answerable"]), key=lambda q: q["question_id"])[::3][: len(unanswerable)]
    return [q for pair in zip(unanswerable, answerable) for q in pair]


def run(subset: list[dict], cached: dict, model: str) -> dict:
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for n, q in enumerate(subset, 1):
        if q["question_id"] in results:
            continue
        passages = [h.chunk["text"] for h in cached[q["question_id"]]["hits"]]
        reply = generate(build_prompt(q["question"], passages), system=SYSTEM, schema=SCHEMA, model=model)
        try:
            parsed = json.loads(reply.text)
        except json.JSONDecodeError:
            parsed = {"quotes": [], "answer": reply.text}
        passages_norm = norm(" ".join(passages))
        quotes = [x.get("text", "") for x in parsed.get("quotes", []) if isinstance(x, dict)]
        verified = [quote_verified(x, passages_norm) for x in quotes]
        results[q["question_id"]] = {
            "quotes": quotes,
            "verified": verified,
            "answer": parsed.get("answer", ""),
            "latency_s": round(reply.latency_s, 1),
            "model": model,
        }
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{n}/{len(subset)} {reply.latency_s:.0f}s quotes={len(quotes)} verified={sum(verified)}", flush=True)
    return results


def auc(positive: list[float], negative: list[float]) -> float:
    """P(a random answerable question scores higher than a random unanswerable one), ties count half."""
    return statistics.mean((p > n) + 0.5 * (p == n) for p in positive for n in negative)


def report(subset: list[dict], results: dict) -> None:
    baseline = json.loads((HERE / "answers.json").read_text(encoding="utf-8"))
    rows = []
    for q in subset:
        r = results.get(q["question_id"])
        if r is None:
            continue
        n_ok = sum(r["verified"])
        refs = [x for x in (q["gold_free_form"], q["gold_augmented"]) if x]
        evidence = [norm(e) for e in q["evidence"]]
        quotes_on_gold = any(norm(x)[:60] in e or e[:60] in norm(x) for x, ok in zip(r["quotes"], r["verified"]) if ok for e in evidence)
        rows.append({
            "answerable": q["answerable"],
            "abstain": n_ok == 0,
            "n_verified": n_ok,
            "n_quotes": len(r["quotes"]),
            "baseline_refused": baseline[q["question_id"]]["text"].strip().lower() == NOT_FOUND.lower(),
            "rouge": max((rouge_l(r["answer"], ref) for ref in refs), default=0.0) if n_ok else 0.0,
            "quotes_on_gold": quotes_on_gold,
            "has_gold_evidence": bool(evidence),
        })
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]

    def rate(group: list[dict], key: str) -> str:
        mean, low, high = bootstrap_ci([float(r[key]) for r in group])
        return f"{sum(r[key] for r in group)} of {len(group)} ({mean:.0%} [{low:.0%}, {high:.0%}])"

    def natural_f1(key: str) -> float:
        """Answerability F1 (unanswerable class) at PeerQA's real mix (46 unanswerable : 148 answerable)."""
        tpr = statistics.mean(float(r[key]) for r in un)
        fpr = statistics.mean(float(r[key]) for r in ans)
        tp, fp = tpr * 46, fpr * 148
        precision = tp / (tp + fp) if tp + fp else 0.0
        return 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0

    total_quotes = sum(r["n_quotes"] for r in rows)
    answered = [r for r in ans if not r["abstain"]]
    with_gold = [r for r in answered if r["has_gold_evidence"]]
    lines = [
        "# PeerQA: quote-then-answer experiment",
        "",
        f"Balanced subset: {len(un)} unanswerable + {len(ans)} answerable questions; same cached top-5 passages as the main run;",
        f"generator {next(iter(results.values()))['model']}. 95% bootstrap intervals. The baseline is the main run's answers on the same questions.",
        "",
        "| Decision rule | Unanswerable abstained | Answerable wrongly abstained | Answerability F1 at real mix |",
        "|---|---|---|---|",
        f"| Baseline: rerank gate + model refusal | {rate(un, 'baseline_refused')} | {rate(ans, 'baseline_refused')} | {natural_f1('baseline_refused'):.2f} |",
        f"| Quote check: no verified quote -> abstain | {rate(un, 'abstain')} | {rate(ans, 'abstain')} | {natural_f1('abstain'):.2f} |",
        "",
        "For reference, refusing everything scores F1 0.38 at the real mix.",
        f"AUC of the verified-quote count (answerable vs unanswerable): {auc([r['n_verified'] for r in ans], [r['n_verified'] for r in un]):.3f}"
        " (the rerank score's was 0.473, the null-QA model's 0.553, on all 194 questions).",
        "",
        f"- Quotes that passed verification: {sum(r['n_verified'] for r in rows)} of {total_quotes}",
        f"- Answered answerable questions whose verified quotes overlap the authors' evidence: {sum(r['quotes_on_gold'] for r in with_gold)} of {len(with_gold)}",
        f"- Rouge-L, answerable questions answered (n={len(answered)}): "
        + (f"{statistics.mean(r['rouge'] for r in answered):.3f}" if answered else "n/a"),
        f"- Mean latency: {statistics.mean(r['latency_s'] for r in results.values()):.1f} s",
    ]
    (HERE / "quote_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    cached = pickle.load(open(HERE / "cache" / "hits.pkl", "rb"))
    subset = pick_subset(questions)
    results = run(subset, cached, args.model)
    report(subset, results)


if __name__ == "__main__":
    main()
