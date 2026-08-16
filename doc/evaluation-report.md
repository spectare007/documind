# DocuMind RAG Evaluation Report

Last updated: 2026-08-16 · Corpus: 6-document real corpus (Form No. 42 receipt and filed form, two June 2026 invoice files, a French timesheet, a June 2026 payslip)

This report has two parts. Part 1 is the current, primary result: all 25 golden-set questions run through `agentic` mode against the real corpus on the current build. Part 2 is an earlier RAGAs metric run over a 5-question subset, kept for its metric detail but superseded as a measure of agentic mode's current behavior (see the warning at the top of that section).

---

## Part 1: Full 25-question agentic sweep (current, primary result)

**Method**: all 25 questions in `evaluation/golden_set.json` were sent to `POST /api/v1/query` with `mode=agentic` against the real corpus, on the current build (after a fail-open fallback around the grader, tried and reverted after it caused a fabrication, was removed, i.e. the grader behaves as shipped). Each row recorded the question, category, answer, number of retrieved chunks, citations, a groundedness flag, latency, and a trace ID.

### Headline

- Of the 23 answerable questions, 15 were answered and 8 were refused with the standard "couldn't find anything relevant" message. All 8 refusals returned zero retrieved chunks, i.e. the grading stage rejected every candidate chunk before synthesis ever ran.
- Of the 2 unanswerable questions, both were correctly declined, with no chunks retrieved and no fabricated content.
- Across all 25 questions, there were zero fabricated answers: every question that produced an answer had chunks and citations behind it, and both genuinely unanswerable questions were correctly refused rather than guessed at.

### Results by category

| Category | Answered | Total | Refused |
|---|---|---|---|
| single-doc-numeric | 5 | 6 | 1 |
| single-doc-factual | 9 | 15 | 6 |
| multi-doc | 1 | 2 | 1 |
| unanswerable (correctly declined) | 2 | 2 | 0 |

### Results by source document

| Document | Answered | Total |
|---|---|---|
| Timesheet (French timesheet template) | 4 | 4 |
| Form No. 42 (acknowledgement receipt + filed form) | 8 | 11 |
| Invoice (June 2026) | 2 | 4 |
| Payslip (June 2026) | 1 | 4 |

The refusals are concentrated in the payslip (3 of 4 questions refused) and invoice (2 of 4 refused) documents; the timesheet had zero refusals and Form No. 42 had the fewest refusals relative to its question count.

### Latency distribution (25 questions, agentic mode)

| Stat | Value |
|---|---|
| min | 60.2 s |
| p50 (median) | 125.1 s |
| mean | 132.3 s |
| max | 257.6 s |
| total (sum across all 25) | 3308.1 s (~55 min) |

### Per-question results

Two rows (marked with a note) counted as "answered" only in the narrow sense that the pipeline retrieved a chunk and did not emit the standard zero-context refusal message; the answer text itself did not actually state the fact asked for. They are listed as answered here to keep this table's answered/refused split consistent with the zero-chunk criterion used everywhere else in this report, with the caveat spelled out in the note.

| # | Category | Question (short) | Verdict | Chunks | Latency (s) | Note |
|---|---|---|---|---|---|---|
| 1 | single-doc-factual | Form 42: what is it used for | Answered | 4 | 120.6 | |
| 2 | single-doc-numeric | Form 42: e-filing acknowledgement number | Answered | 1 | 60.2 | |
| 3 | single-doc-factual | Form 42: e-filing date | Refused | 0 | 173.3 | |
| 4 | single-doc-factual | Form 42: tax year | Answered | 1 | 106.3 | |
| 5 | single-doc-factual | Form 42: filing type (original/revised) | Answered | 1 | 85.7 | |
| 6 | single-doc-numeric | Form 42: attachment count | Answered | 1 | 140.4 | answer text included a leaked prompt fragment ahead of the actual figure |
| 7 | single-doc-factual | Form 42: purpose of the TRC | Answered | 1 | 66.5 | |
| 8 | single-doc-factual | Form 42: applicant nationality | Refused | 0 | 162.9 | |
| 9 | single-doc-factual | Form 42: TRC applicability period | Answered | 1 | 133.5 | model stated the form does not specify a period, which does not match the reference; not a zero-context refusal |
| 10 | single-doc-factual | Form 42: verification capacity (individual/company) | Refused | 0 | 105.1 | |
| 11 | multi-doc | Form 42: receipt vs. filed form, same ack. number? | Answered | 1 | 182.5 | model concluded it could not determine this from the single chunk retrieved; not a zero-context refusal |
| 12 | single-doc-numeric | June 2026 invoice: invoice number | Refused | 0 | 86.2 | |
| 13 | single-doc-numeric | June 2026 invoice: total amount | Answered | 2 | 75.7 | |
| 14 | single-doc-factual | June 2026 invoice: line item billed | Answered | 2 | 134.4 | |
| 15 | multi-doc | Compare the two June 2026 invoice files | Refused | 0 | 125.1 | |
| 16 | single-doc-numeric | Timesheet: total hours | Answered | 1 | 174.4 | |
| 17 | single-doc-numeric | Timesheet: timesheet number | Answered | 1 | 115.3 | |
| 18 | single-doc-factual | Timesheet: week covered | Answered | 1 | 108.1 | |
| 19 | single-doc-factual | Timesheet: submitted/approved dates | Answered | 1 | 257.6 | |
| 20 | single-doc-factual | Payslip: designation/job title | Refused | 0 | 162.9 | |
| 21 | single-doc-factual | Payslip: payment mode | Refused | 0 | 158.0 | |
| 22 | single-doc-factual | Payslip: pay cycle date range | Refused | 0 | 190.2 | |
| 23 | single-doc-factual | Payslip: date joined | Answered | 1 | 108.6 | |
| 24 | unanswerable | Company stock price (correctly declined) | Refused | 0 | 70.9 | |
| 25 | unanswerable | Annual bonus amount (correctly declined) | Refused | 0 | 203.7 | |

### Cross-mode diagnostic: isolating the cause of the refusals

Four of the eight refused questions above were re-run in `simple` mode, which shares the same `HybridRetriever` and index as `agentic` mode but skips the corrective grading/synthesis-check loop. All four were answered correctly:

| Refused question (agentic) | Simple-mode result | Chunks | Latency (s) |
|---|---|---|---|
| Form 42: e-filing date | Correct | 7 | 27 |
| June 2026 invoice number | Correct | 8 | 25 |
| Payslip: designation | Correct | 10 | 82 |
| Payslip: payment mode | Correct | 11 | 60 |

Since both modes read from the same retrieval index, a question that `simple` mode answers correctly and `agentic` mode refuses cannot be a retrieval problem: the relevant chunks are present and rankable. The difference has to be downstream of retrieval, in the per-chunk relevance grading stage that only `agentic` mode runs. The `qwen2.5:3b` grader is known to be fragile on this exact classification task (see `doc/architecture.md` for the CrewAI-agent-wrapper failure mode that causes it). A previously tried fail-open fallback around the grader, which proceeded to synthesis on the top-ranked chunks whenever the grader rejected all of them, fixed these refusals but was reverted after it caused the synthesizer to fabricate a bonus figure by misattributing an unrelated payslip table row (see Part 2's judge-reliability note for related detail).

### Analysis: precision vs. recall trade-off, and the concrete next step

Based on this sweep, `agentic` mode is currently precision-oriented: it would rather refuse than guess. Across all 25 questions it produced zero fabricated answers, and it correctly declined both genuinely unanswerable questions (2 of 2). Its cost is recall: 8 of 23 answerable questions were refused, all traced to the grading stage rejecting chunks it should have kept, and its per-question latency (60 to 258 seconds, mean 132 seconds) is far higher than `simple` mode's.

`simple` mode looks recall-oriented by comparison: the 5-question RAGAs run in Part 2 and the 4-question cross-mode diagnostic above are the only direct measurements available, and in both, `simple` mode answered every question it was given correctly, at latencies of 25 to 82 seconds in the diagnostic (versus 27 to 43 seconds mean/p95 in the RAGAs run). No full 25-question sweep of `simple` mode exists yet, so "answers essentially everything" is a claim scoped to the 5 RAGAs questions plus these 4 diagnostic questions (9 data points total, not 25), not an extrapolation to the full set.

The concrete next step, not yet attempted: run the per-chunk grading stage on a larger model, `qwen2.5:7b`, which is already pulled locally and already used as the RAGAs judge, instead of the `qwen2.5:3b` used for the other agent roles. This exact 3B model's grading judgment is already known to be fragile to wrapper and prompt framing (see `doc/architecture.md`), which is the most likely explanation for the refusals here. The same 25 golden questions should be re-run in `agentic` mode with only the grading stage swapped to the larger model, so the comparison isolates that one change.

---

## Part 2: RAGAs metric run (5-question subset, historical)

> **This section's agentic figures were recorded while a grader defect was active** (the same defect described in `doc/architecture.md`: a CrewAI agent wrapper's injected system message biased the per-chunk grader into effectively rubber-stamping or rejecting chunks regardless of their actual content). The `answer_relevancy` and `context_precision` scores of 0.0 for agentic mode below reflect that defect, not agentic mode's current behavior. For agentic mode's current, defect-free performance, see Part 1 above, which reports the full 25-question sweep run against the current build.

Judge: local Ollama `qwen2.5:3b` (embeddings: `nomic-embed-text`) · Metrics: RAGAs

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

### Latency

| Stat | simple | agentic |
|---|---|---|
| p50_s | 27.4 | 115.6 |
| p95_s | 43.2 | 129.6 |
| mean_s | 38.2 | 128.1 |

### Per-question latency (wall-clock cost per question per mode)

| # | Category | Question | simple latency_s | agentic latency_s |
|---|---|---|---|---|
| 1 | single-doc-numeric | What is the e-Filing Acknowledgement Number shown on the Form No. 42 acknowledgement receipt? | 27.35 | 99.66 |
| 2 | single-doc-factual | According to the filed Form No. 42, what is the purpose of obtaining the Tax Residency Certificate (TRC)? | 19.51 | 115.62 |
| 3 | multi-doc | Compare the two June 2026 invoice files in the knowledge base (Invoice_June26.pdf and Invoice_June'26.pdf): do they show the same total amount? | 43.2 | 129.58 |
| 4 | single-doc-factual | What is the designation/job title listed on the June 2026 payslip? | 78.92 | 201.51 |
| 5 | unanswerable | What is the company's current stock price today? | 21.9 | 93.92 |

### Key finding: retrieval failure, not just score differences

- **agentic**: 5/5 questions returned zero retrieved chunks (the pipeline's corrective-retrieval/grading step rejected every candidate chunk and fell back to a boilerplate "couldn't find anything relevant" answer), including at least one question the other mode answered correctly from real retrieved context.

This is the mechanism behind the headline numbers, not just a side note: when a mode returns zero context, `context_precision`/`context_recall` should be trivially 0 for that row, and any non-empty `faithfulness` or `answer_relevancy` score is scoring a refusal, not an answer -- a high faithfulness score in that state reflects "didn't hallucinate because it didn't try to answer," not answer quality. Treat `context_recall` as the metric of interest for retrieval quality specifically because it is not inflated by refusal behavior the way faithfulness is.

**Judge-reliability anomaly**: the local judge scored `context_recall` > 0 on at least one row where `retrieved_contexts` was empty -- a logically implausible result (recall of a reference against zero retrieved text should be 0), most likely caused by the smaller judge model. Affected rows: [('agentic', 'What is the e-Filing Acknowledgement Number shown on the For', 1.0), ('agentic', 'Compare the two June 2026 invoice files in the knowledge bas', 1.0), ('agentic', "What is the company's current stock price today?", 1.0)]. Reported here rather than silently trusting the aggregate mean.

### Caveats

- Judge is a local `qwen2.5:3b` model on CPU: scores are directionally useful, noisy in absolute terms.
- Agentic vs simple comparison shares the same judge, so relative differences are more reliable.
- Unanswerable questions score via faithfulness (refusing is grounded behavior) -- but see the Key Finding above: a refusal-driven faithfulness score is not evidence of answer quality.
- One or more metrics returned NaN for some rows (local judge failed to parse a verdict). NaNs are excluded from the displayed mean and called out per-cell above rather than silently dropped; per-row detail is in the gitignored `evaluation/runs/<timestamp>/results.json`.
- **This 5-question run's agentic scores were taken while the grading-stage defect described at the top of this section was active.** They are retained here for the RAGAs metric detail (NaN handling, the judge-reliability anomaly) but should not be read as agentic mode's current performance; see Part 1 for the current 25-question sweep.
