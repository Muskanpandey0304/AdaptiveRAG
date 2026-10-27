# Embedding model comparison (retrieval only)

## dev (49 in-scope questions)

| Model | dim | embed time (s) | vector hit@5 | vector MRR | hybrid hit@5 | hybrid MRR |
|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 592 | 0.796 | 0.554 | 0.837 | 0.594 |
| intfloat/e5-base-v2 | 768 | 1932 | 0.816 | 0.567 | 0.857 | 0.655 |
| BAAI/bge-base-en-v1.5 | 768 | 1918 | 0.735 | 0.540 | 0.837 | 0.581 |

## held-out (27 in-scope questions)

| Model | dim | embed time (s) | vector hit@5 | vector MRR | hybrid hit@5 | hybrid MRR |
|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 592 | 0.741 | 0.512 | 0.815 | 0.633 |
| intfloat/e5-base-v2 | 768 | 1932 | 0.926 | 0.625 | 0.889 | 0.728 |
| BAAI/bge-base-en-v1.5 | 768 | 1918 | 0.852 | 0.600 | 0.815 | 0.597 |

## Similarity distributions (for re-tuning VECTOR_MIN_SIM / CENTROID_MIN_SIM)

### BAAI/bge-small-en-v1.5
- dev scoped: best-chunk cosine min 0.724 / p10 0.761 / median 0.827 / max 0.891
- dev scoped: centroid cosine min 0.596 / p10 0.628 / median 0.674 / max 0.769
- dev general: best-chunk cosine min 0.576 / p10 0.578 / median 0.617 / max 0.698
- dev general: centroid cosine min 0.461 / p10 0.471 / median 0.521 / max 0.610
- held-out scoped: best-chunk cosine min 0.727 / p10 0.763 / median 0.810 / max 0.895
- held-out scoped: centroid cosine min 0.608 / p10 0.618 / median 0.675 / max 0.729
- held-out general: best-chunk cosine min 0.543 / p10 0.581 / median 0.651 / max 0.788
- held-out general: centroid cosine min 0.462 / p10 0.483 / median 0.565 / max 0.711

### intfloat/e5-base-v2
- dev scoped: best-chunk cosine min 0.852 / p10 0.861 / median 0.891 / max 0.936
- dev scoped: centroid cosine min 0.808 / p10 0.841 / median 0.857 / max 0.880
- dev general: best-chunk cosine min 0.756 / p10 0.765 / median 0.778 / max 0.831
- dev general: centroid cosine min 0.762 / p10 0.763 / median 0.788 / max 0.812
- held-out scoped: best-chunk cosine min 0.836 / p10 0.859 / median 0.896 / max 0.916
- held-out scoped: centroid cosine min 0.832 / p10 0.842 / median 0.855 / max 0.873
- held-out general: best-chunk cosine min 0.743 / p10 0.778 / median 0.803 / max 0.880
- held-out general: centroid cosine min 0.757 / p10 0.780 / median 0.804 / max 0.863

### BAAI/bge-base-en-v1.5
- dev scoped: best-chunk cosine min 0.665 / p10 0.701 / median 0.765 / max 0.900
- dev scoped: centroid cosine min 0.502 / p10 0.548 / median 0.601 / max 0.712
- dev general: best-chunk cosine min 0.505 / p10 0.521 / median 0.550 / max 0.601
- dev general: centroid cosine min 0.320 / p10 0.377 / median 0.452 / max 0.550
- held-out scoped: best-chunk cosine min 0.666 / p10 0.709 / median 0.763 / max 0.844
- held-out scoped: centroid cosine min 0.521 / p10 0.534 / median 0.590 / max 0.650
- held-out general: best-chunk cosine min 0.495 / p10 0.500 / median 0.575 / max 0.742
- held-out general: centroid cosine min 0.415 / p10 0.424 / median 0.489 / max 0.643
