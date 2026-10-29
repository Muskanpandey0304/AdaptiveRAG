# PeerQA (outside benchmark, answer quality and answerability)

[PeerQA](https://arxiv.org/abs/2502.13668) (Baumgartner et al., NAACL 2025): questions that peer reviewers asked about a
paper, answered by the paper's authors, with the evidence sentences they highlighted and a flag for whether the paper
can answer the question at all. Licence CC-BY-NC-SA 4.0. This folder runs the project's own retrieval and answer path
on the 90 papers whose licences allow redistribution (194 labelled questions: 148 answerable, 46 unanswerable). Each
question searches only its own paper. Generator: llama3.1:8b. Nothing here is tuned on the data.

**Read this before the numbers: they are weak, and the cause is understood.** The pipeline's rerank-confidence gate
(`RERANK_MIN_PROB = 0.5`) refuses 62% of the answerable questions. That gate was built and tuned to catch off-topic
questions across our own 26 papers, and it does that well. It does not transfer to PeerQA: inside one paper every
passage is on topic, so the rerank score cannot tell "the answer is here" from "it is not" (AUC 0.47 for answerable vs
unanswerable, a coin flip). Reviewer-style questions ("Why was X not compared?") also score low on relevance even when the
paper answers them. So the headline answer score (Rouge-L 0.09 over all answerable questions) mostly measures refusals,
not answer quality; among questions the system did answer it is 0.24, a rough figure because Rouge-L only counts word
overlap.

Fixes tried for the answerability decision, none adopted:

| Idea | Result |
|---|---|
| Different rerank threshold | Not a fix: no threshold separates the groups (AUC 0.47) |
| Small SQuAD2-style QA model's "no answer" score | AUC 0.553, within noise on 46 unanswerable questions |
| Quote-then-answer (model copies supporting sentences, code checks they occur in the passages) | AUC about 0.46 on 56 questions; the model finds real but irrelevant sentences, so verified quotes prove faithfulness, not answerability. About 94% of quotes were genuine text from the passages |
| Skipping the gate when the paper is already known ("paper-scoped mode") | Not run. The model's own refusals do carry some signal (it refused 15 of 25 unanswerable and 25 of 81 answerable questions that passed the gate) |

Evidence retrieval is moderate: at least one gold evidence sentence is in the top 5 passages for 43% of questions (the
authors' best retrievers reach about 0.48 MRR and 0.68 Recall@10 on their own units, so the numbers are not directly
comparable).

## Files

- `results.md`, `results.json`: full-run scores. `retrieval_results.json`: retrieval scores per question.
- `quote_results.md`, `quote_check.py`: the quote-then-answer experiment (stopped after 56 of 92 planned questions; resumable).
- `answers.json`, `quote_answers.json`, `questions.json`: model answers and the prepared questions. These copy PeerQA text
  (CC BY-NC-SA), so they are gitignored; `prepare.py` recreates `questions.json`.
- `prepare.py`, `retrieve.py`, `generate.py`, `score.py`: the pipeline, run in that order.
- `data/`, `index/`, `cache/` are local and gitignored. The dataset zip must be downloaded by hand from
  https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/peerqa-data-v1.0.zip?sequence=1&isAllowed=y
  (the server shows a bot check to scripts) and unzipped into `data/`.

```
PYTHONPATH=. python benchmarks/peerqa/prepare.py
EMBED_DEVICE=cuda PYTHONPATH=. venv_gpu/Scripts/python.exe benchmarks/peerqa/retrieve.py
PYTHONPATH=. venv/Scripts/python.exe benchmarks/peerqa/generate.py
PYTHONPATH=. python benchmarks/peerqa/score.py
```
