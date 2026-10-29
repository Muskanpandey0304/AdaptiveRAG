# PeerQA results

194 questions on 90 papers (148 answerable, 46 unanswerable);
each question searches only its own paper. Generator: llama3.1:8b. 95% bootstrap intervals over questions.
No human or LLM judging: Rouge-L is word overlap with the author's answer (see score.py).

## Retrieval (136 answerable questions with mapped evidence)

- At least one gold evidence sentence in the top 5 passages, hybrid: 0.419 [0.338, 0.507]
- At least one gold evidence sentence in the top 5 passages, hybrid + rerank: 0.426 [0.346, 0.507]
- Share of gold evidence sentences in the top 5 passages, hybrid + rerank: 0.238 [0.184, 0.294]

## Answers

| Measure | Value |
|---|---|
| Unanswerable questions correctly refused | 36 of 46 (0.783 [0.652, 0.891]) |
| Answerable questions wrongly refused | 92 of 148 (0.622 [0.547, 0.703]) |
| Answerability F1 (unanswerable class), precision, recall | 0.414, 0.281, 0.783 |
| Rouge-L, all answerable (refusal = 0) | 0.091 [0.063, 0.122] |
| Rouge-L, answerable questions the system answered (n=56) | 0.240 [0.184, 0.302] |

Mean latency when the model generated: 44.9 s.
