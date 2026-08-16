# DocuMind RAG Evaluation Report

Generated: 2026-08-16 09:28 UTC · Judge: local Ollama `qwen2.5:3b` (embeddings: `nomic-embed-text`) · Metrics: RAGAs

**Judge substitution**: this run used `qwen2.5:3b` instead of the originally planned `qwen2.5:7b`, to fit the available time budget on a CPU-only host. A smaller judge further weakens the precision of the absolute scores below (see Caveats) -- relative agentic-vs-simple comparisons remain the more trustworthy read since both share the same judge.

**Reduced run**: 5 of 25 golden-set questions were evaluated in this recorded run, in **both** modes. The reduction is a CPU-time constraint (a local judge plus a full six-agent crewai crew per agentic question), not a methodological choice -- a deliberately stratified subset was selected, not the first N. An initial attempt at 10 questions per mode with the qwen2.5:7b judge was aborted after observing a ~2 hour remaining-time projection from the judging pace; the judge was then swapped to qwen2.5:3b and the subset cut to 5 questions per mode to fit the available time budget. The subset was chosen deliberately to span categories (single-doc factual/numeric, multi-doc, unanswerable) and all source documents, not truncated to the first N. The full golden set stays committed at `evaluation/golden_set.json`.

Questions evaluated in this run:

1. [single-doc-numeric] What is the e-Filing Acknowledgement Number shown on the Form No. 42 acknowledgement receipt?
2. [single-doc-factual] According to the filed Form No. 42, what is the purpose of obtaining the Tax Residency Certificate (TRC)?
3. [multi-doc] Compare the two June 2026 invoice files in the knowledge base (Invoice_June26.pdf and Invoice_June'26.pdf): do they show the same total amount?
4. [single-doc-factual] What is the designation/job title listed on the June 2026 payslip?
5. [unanswerable] What is the company's current stock price today?

| Metric | simple | agentic |
|---|---|---|
| faithfulness | 0.5 (1/5 NaN) | 0.867 |
| answer_relevancy | 0.571 | 0.0 |
| context_precision | 0.49 | 0.0 |
| context_recall | 1.0 (2/5 NaN) | 0.6 |

## Latency

| Stat | simple | agentic |
|---|---|---|
| p50_s | 27.4 | 115.6 |
| p95_s | 43.2 | 129.6 |
| mean_s | 38.2 | 128.1 |

## Per-question latency (wall-clock cost per question per mode)

| # | Category | Question | simple latency_s | agentic latency_s |
|---|---|---|---|---|
| 1 | single-doc-numeric | What is the e-Filing Acknowledgement Number shown on the Form No. 42 acknowledgement receipt? | 27.35 | 99.66 |
| 2 | single-doc-factual | According to the filed Form No. 42, what is the purpose of obtaining the Tax Residency Certificate (TRC)? | 19.51 | 115.62 |
| 3 | multi-doc | Compare the two June 2026 invoice files in the knowledge base (Invoice_June26.pdf and Invoice_June'26.pdf): do they show the same total amount? | 43.2 | 129.58 |
| 4 | single-doc-factual | What is the designation/job title listed on the June 2026 payslip? | 78.92 | 201.51 |
| 5 | unanswerable | What is the company's current stock price today? | 21.9 | 93.92 |

## Key finding: retrieval failure, not just score differences

- **agentic**: 5/5 questions returned zero retrieved chunks (the pipeline's corrective-retrieval/grading step rejected every candidate chunk and fell back to a boilerplate "couldn't find anything relevant" answer), including at least one question the other mode answered correctly from real retrieved context.

This is the mechanism behind the headline numbers, not just a side note: when a mode returns zero context, `context_precision`/`context_recall` should be trivially 0 for that row, and any non-empty `faithfulness` or `answer_relevancy` score is scoring a refusal, not an answer -- a high faithfulness score in that state reflects "didn't hallucinate because it didn't try to answer," not answer quality. Treat `context_recall` as the metric of interest for retrieval quality specifically because it is not inflated by refusal behavior the way faithfulness is.

**Judge-reliability anomaly**: the local judge scored `context_recall` > 0 on at least one row where `retrieved_contexts` was empty -- a logically implausible result (recall of a reference against zero retrieved text should be 0), most likely caused by the smaller judge model. Affected rows: [('agentic', 'What is the e-Filing Acknowledgement Number shown on the For', 1.0), ('agentic', 'Compare the two June 2026 invoice files in the knowledge bas', 1.0), ('agentic', "What is the company's current stock price today?", 1.0)]. Reported here rather than silently trusting the aggregate mean.

## Caveats

- Judge is a local `qwen2.5:3b` model on CPU: scores are directionally useful, noisy in absolute terms.
- Agentic vs simple comparison shares the same judge, so relative differences are more reliable.
- Unanswerable questions score via faithfulness (refusing is grounded behavior) -- but see the Key Finding above: a refusal-driven faithfulness score is not evidence of answer quality.
- One or more metrics returned NaN for some rows (local judge failed to parse a verdict). NaNs are excluded from the displayed mean and called out per-cell above rather than silently dropped; per-row detail is in the gitignored `evaluation/runs/<timestamp>/results.json`.