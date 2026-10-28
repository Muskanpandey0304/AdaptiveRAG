# SciFact (BEIR) results

5183 abstracts, 300 test claims. Same retrieval components and settings as the main
pipeline (intfloat/e5-base-v2, BM25, RRF, BAAI/bge-reranker-base); abstracts indexed whole. nDCG@10 / Recall@10 with
95% bootstrap intervals over queries.

| Method | nDCG@10 | 95% CI | Recall@10 |
|---|---|---|---|
| BM25 | 0.652 | [0.605, 0.697] | 0.774 |
| Dense (e5-base-v2) | 0.719 | [0.678, 0.759] | 0.846 |
| Hybrid (RRF) | 0.727 | [0.686, 0.767] | 0.864 |
| Hybrid + rerank | 0.730 | [0.689, 0.769] | 0.867 |
