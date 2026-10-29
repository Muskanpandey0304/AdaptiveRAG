"""Build the PeerQA benchmark inputs from the dataset zip (data/peerqa-data-v1.0.zip, unzipped into data/).

PeerQA (Baumgartner et al., NAACL 2025) has questions that peer reviewers asked about a paper and answers
written by the paper's authors, with the evidence sentences the authors highlighted and an answerable flag.
The download holds the extracted text of the 90 papers with redistributable licences (the other papers need
OpenReview downloads and are left out). Each paper becomes one Markdown file (one page) so the project's own
loader, chunker and parent/child splitter run on it unchanged. Questions without an answerable label are dropped.

    PYTHONPATH=. python benchmarks/peerqa/prepare.py
"""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
MD_DIR = DATA / "papers_md"


def paper_key(paper_id: str) -> str:
    """File-safe id: 'nlpeer/ARR-22/123' -> 'nlpeer__ARR-22__123'."""
    return paper_id.replace("/", "__")


def paper_markdown(rows: list[dict]) -> str:
    """Title, then headings and paragraphs in reading order. Sentences of one paragraph (same pidx) are joined."""
    lines: list[str] = []
    title_done = False
    current_pidx, buffer = None, []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            lines.extend([" ".join(buffer), ""])
            buffer = []

    for row in sorted(rows, key=lambda r: r["idx"]):
        text = row["content"].strip()
        if not text:
            continue
        if row["type"] == "title" and not title_done:
            lines.extend([f"# {text}", "", "<!-- page: 1 -->", ""])
            title_done = True
        elif row["type"] == "heading":
            flush()
            lines.extend([f"## {text}", ""])
        elif row["type"] == "sentence":
            if row["pidx"] != current_pidx:
                flush()
                current_pidx = row["pidx"]
            buffer.append(text)
        else:  # table, figure, formula, list_item: one block each
            flush()
            lines.extend([text, ""])
    flush()
    return "\n".join(lines)


def evidence_sentences(question: dict) -> list[str]:
    """The author-highlighted evidence that could be mapped onto the extracted text."""
    mapped = question.get("answer_evidence_mapped") or []
    if isinstance(mapped, dict):
        mapped = [mapped]
    sentences: list[str] = []
    for item in mapped:
        found = item.get("sentence") if isinstance(item, dict) else None
        sentences.extend([found] if isinstance(found, str) else (found or []))
    return [s for s in sentences if s.strip()]


def main() -> None:
    papers = [json.loads(line) for line in open(DATA / "papers.jsonl", encoding="utf-8")]
    questions = [json.loads(line) for line in open(DATA / "qa.jsonl", encoding="utf-8")]
    augmented = {
        row["question_id"]: row["augmented_answer_free_form"]
        for row in map(json.loads, open(DATA / "qa-augmented-answers.jsonl", encoding="utf-8"))
    }

    rows_by_paper: dict[str, list[dict]] = defaultdict(list)
    for row in papers:
        rows_by_paper[row["paper_id"]].append(row)
    MD_DIR.mkdir(parents=True, exist_ok=True)
    for paper_id, rows in rows_by_paper.items():
        (MD_DIR / f"{paper_key(paper_id)}.md").write_text(paper_markdown(rows), encoding="utf-8")

    kept = []
    for q in questions:
        if q["paper_id"] not in rows_by_paper or q["answerable"] is None:
            continue
        kept.append(
            {
                "question_id": q["question_id"],
                "paper_id": paper_key(q["paper_id"]),
                "venue": "/".join(q["paper_id"].split("/")[:2]),
                "question": q["question"],
                "answerable": bool(q["answerable"]),
                "gold_free_form": q.get("answer_free_form") or "",
                "gold_augmented": augmented.get(q["question_id"]) or "",
                "evidence": evidence_sentences(q) if q["answerable"] else [],
            }
        )
    (HERE / "questions.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
    answerable = sum(q["answerable"] for q in kept)
    with_evidence = sum(bool(q["evidence"]) for q in kept)
    with_gold = sum(bool(q["gold_free_form"] or q["gold_augmented"]) for q in kept if q["answerable"])
    print(
        f"{len(rows_by_paper)} papers, {len(kept)} questions: {answerable} answerable, {len(kept) - answerable} unanswerable; "
        f"{with_evidence} with mapped evidence, {with_gold} answerable with a gold answer"
    )


if __name__ == "__main__":
    main()
