# PeerQA: quote-then-answer experiment

Balanced subset: 28 unanswerable + 28 answerable questions; same cached top-5 passages as the main run;
generator llama3.1:8b. 95% bootstrap intervals. The baseline is the main run's answers on the same questions.

| Decision rule | Unanswerable abstained | Answerable wrongly abstained | Answerability F1 at real mix |
|---|---|---|---|
| Baseline: rerank gate + model refusal | 22 of 28 (79% [61%, 93%]) | 14 of 28 (50% [32%, 68%]) | 0.46 |
| Quote check: no verified quote -> abstain | 5 of 28 (18% [4%, 36%]) | 10 of 28 (36% [18%, 54%]) | 0.15 |

For reference, refusing everything scores F1 0.38 at the real mix.
AUC of the verified-quote count (answerable vs unanswerable): 0.460 (the rerank score's was 0.473, the null-QA model's 0.553, on all 194 questions).

- Quotes that passed verification: 96 of 102
- Answered answerable questions whose verified quotes overlap the authors' evidence: 9 of 18
- Rouge-L, answerable questions answered (n=18): 0.279
- Mean latency: 82.7 s

Stopped after 56 of the planned 92 questions (28 + 28, still balanced): the quote check showed no signal, and the remaining hour would not change that. The script is resumable.
