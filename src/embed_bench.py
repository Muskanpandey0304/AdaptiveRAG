"""Compare embedding models on retrieval quality alone (no LLM, fast to re-run).

    python -m src.embed_bench                       # all default candidates
    python -m src.embed_bench BAAI/bge-base-en-v1.5 # just one

For each model it embeds the current child chunks (data/index/chunks.json) and
every in-scope question in eval/questions.json + eval/heldout_questions.json,
then reports, per question set:
  * hit@5 and MRR for vector-only search and for hybrid (BM25 + vector, RRF),
    judged exactly as src.eval does: children collapse to their parents, the
    top TOP_K parents are kept, and a hit is a parent on the gold source+page;
  * the best-chunk cosine and query-vs-corpus-centroid cosine for in-scope and
    off-topic questions. VECTOR_MIN_SIM and CENTROID_MIN_SIM in config.py are
    absolute cosines tuned for one model, so after switching models, pick them
    from these distributions (in-scope min/p10 vs. off-topic max).

Writes eval/embeddings.md. Query/passage templates come from
config.EMBED_PREFIXES, so each model is used the way it was trained.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CANDIDATES_K, EMBED_PREFIXES, TOP_K
from src.indexer import CHUNKS_PATH, build_bm25_index, tokenize
from src.retrievers import _fuse_rrf

QUESTION_FILES = {"dev": Path("eval/questions.json"), "held-out": Path("eval/heldout_questions.json")}
OUT_PATH = Path("eval/embeddings.md")
DEFAULT_MODELS = [
    "BAAI/bge-small-en-v1.5",  # current
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]


def _parent_ranking(child_ids: list[int], children: list[dict]) -> list[int]:
    """Collapse a ranked child list to ranked unique parent ids (best child
    wins the parent's position), the way Retriever.to_parents does."""
    seen: dict[int, None] = {}
    for cid in child_ids:
        seen.setdefault(children[cid]["parent_id"], None)
    return list(seen)


def _first_gold_rank(parent_ids: list[int], parents: list[dict], question: dict) -> int | None:
    """1-based rank of the first of the top-TOP_K parents on the gold page, else None."""
    pages = set(question["gold_pages"] or [])
    for rank, pid in enumerate(parent_ids[:TOP_K], 1):
        if parents[pid]["source"] == question["gold_source"] and parents[pid]["page"] in pages:
            return rank
    return None


def _score(ranks: list[int | None]) -> tuple[float, float]:
    """(hit@TOP_K, MRR) from first-gold ranks (None = miss)."""
    return (
        float(np.mean([r is not None for r in ranks])),
        float(np.mean([1 / r if r else 0.0 for r in ranks])),
    )


def _dist(values: list[float]) -> str:
    return f"min {min(values):.3f} / p10 {np.percentile(values, 10):.3f} / median {np.median(values):.3f} / max {max(values):.3f}"


def evaluate_model(model_name: str, children: list[dict], parents: list[dict], bm25_ranked: dict) -> dict:
    """Embed corpus + questions with `model_name`; return metrics and timing."""
    query_prefix, doc_prefix = EMBED_PREFIXES.get(model_name, ("", ""))
    model = SentenceTransformer(model_name)
    model.max_seq_length = 256  # children are ~100 tokens; avoids pointless padding

    start = time.perf_counter()
    embeddings = model.encode(
        [doc_prefix + c["text"] for c in children], normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    embed_seconds = time.perf_counter() - start
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    result: dict = {"model": model_name, "dim": embeddings.shape[1], "embed_s": embed_seconds, "sets": {}}
    for set_name, questions in bm25_ranked.items():
        scoped = [q for q in questions if q["type"] != "general"]
        general = [q for q in questions if q["type"] == "general"]
        q_vecs = model.encode([query_prefix + q["question"] for q in questions], normalize_embeddings=True)
        sims = q_vecs @ embeddings.T
        centroid_sims = q_vecs @ centroid

        vector_ranks, hybrid_ranks = [], []
        top_cos = {"scoped": [], "general": []}
        centroid_cos = {"scoped": [], "general": []}
        for i, question in enumerate(questions):
            order = np.argsort(sims[i])[::-1][:CANDIDATES_K]
            kind = "general" if question["type"] == "general" else "scoped"
            top_cos[kind].append(float(sims[i][order[0]]))
            centroid_cos[kind].append(float(centroid_sims[i]))
            if kind == "general":
                continue
            vector_ranks.append(_first_gold_rank(_parent_ranking(list(order), children), parents, question))
            fused = _fuse_rrf([[int(c) for c in order], question["_bm25"]])
            hybrid_order = sorted(fused, key=lambda cid: fused[cid], reverse=True)
            hybrid_ranks.append(_first_gold_rank(_parent_ranking(hybrid_order, children), parents, question))

        result["sets"][set_name] = {
            "n": len(scoped),
            "vector": _score(vector_ranks),
            "hybrid": _score(hybrid_ranks),
            "top_cos": {k: _dist(v) for k, v in top_cos.items() if v},
            "centroid_cos": {k: _dist(v) for k, v in centroid_cos.items() if v},
            "n_general": len(general),
        }
    return result


def render(results: list[dict]) -> str:
    """Markdown report: one comparison table per question set, then the
    similarity distributions needed to re-tune the cosine thresholds."""
    lines = ["# Embedding model comparison (retrieval only)", ""]
    for set_name in results[0]["sets"]:
        n = results[0]["sets"][set_name]["n"]
        lines += [
            f"## {set_name} ({n} in-scope questions)",
            "",
            "| Model | dim | embed time (s) | vector hit@5 | vector MRR | hybrid hit@5 | hybrid MRR |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results:
            s = r["sets"][set_name]
            lines.append(
                f"| {r['model']} | {r['dim']} | {r['embed_s']:.0f} | {s['vector'][0]:.3f} | {s['vector'][1]:.3f} "
                f"| {s['hybrid'][0]:.3f} | {s['hybrid'][1]:.3f} |"
            )
        lines.append("")
    lines += ["## Similarity distributions (for re-tuning VECTOR_MIN_SIM / CENTROID_MIN_SIM)", ""]
    for r in results:
        lines.append(f"### {r['model']}")
        for set_name, s in r["sets"].items():
            for kind in ("scoped", "general"):
                if kind in s["top_cos"]:
                    lines.append(f"- {set_name} {kind}: best-chunk cosine {s['top_cos'][kind]}")
                    lines.append(f"- {set_name} {kind}: centroid cosine {s['centroid_cos'][kind]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: python -m src.embed_bench [model ...]."""
    models = sys.argv[1:] or DEFAULT_MODELS
    children = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    parents = json.loads(CHUNKS_PATH.with_name("parents.json").read_text(encoding="utf-8"))
    bm25 = build_bm25_index(children)

    question_sets = {}
    for name, path in QUESTION_FILES.items():
        questions = json.loads(path.read_text(encoding="utf-8"))
        for q in questions:
            # BM25 does not depend on the embedding model: rank once, reuse for every model.
            scores = bm25.get_scores(tokenize(q["question"]))
            q["_bm25"] = [int(i) for i in np.argsort(scores)[::-1][:CANDIDATES_K]]
        question_sets[name] = questions

    results = []
    for model_name in models:
        print(f"=== {model_name}", flush=True)
        results.append(evaluate_model(model_name, children, parents, question_sets))
        OUT_PATH.write_text(render(results), encoding="utf-8")  # saved after each model, so a crash loses nothing
    print(render(results))


if __name__ == "__main__":
    main()
