# DocuMind RAG Evaluation Report

Last updated: 2026-08-17 · Corpus: 6-document real corpus (Form No. 42 receipt and filed form, two June 2026 invoice files, a French timesheet, a June 2026 payslip)

This report has two parts. Part 1 is the current, primary result: all 25 golden-set questions run through `agentic` mode against the real corpus, scored for correctness against their golden references. Part 2 is an earlier RAGAs metric run over a 5-question subset, kept for its metric detail but superseded as a measure of agentic mode's current behavior (see the warning at the top of that section).

Note on values: the corpus is a set of the author's real personal documents, so the committed golden set uses placeholders (`<REDACTED-...>`) in place of the actual figures, dates and identifiers. This report follows the same rule. Answers are characterised, never quoted with their values.

---

## Part 1: Full 25-question agentic sweep, scored for correctness (current, primary result)

**Method**: all 25 questions in `evaluation/golden_set.json` were sent to `POST /api/v1/query` with `mode=agentic` against the real corpus, in a single recorded run on 2026-08-16, on the build of that date (after a fail-open fallback around the grader, tried and reverted after it caused a fabrication, was removed, i.e. the grader behaves as shipped). Each row recorded the question, category, answer, number of retrieved chunks, citations, a groundedness flag, latency, and a trace ID. Every answer was then scored against its golden reference on the rubric below.

### Scoring rubric and how it was applied

An earlier version of this section reported "15 of 23 answered" as the headline. That number was measuring the wrong thing: it counted any response that was not the standard zero-chunk refusal, so a right answer and a wrong answer scored identically. Every answer has since been read against its golden reference and scored on a four-way rubric:

| Verdict | Meaning |
|---|---|
| **correct** | States the fact the reference states, from the right document. |
| **partial** | Right document and topic, but incomplete, garbled, hedged into uselessness, or carrying a visible prompt-leak artifact in the answer text. |
| **refused** | Returned the standard "couldn't find anything relevant" message with zero retrieved chunks. Correct behaviour on an unanswerable question, a recall failure on an answerable one. |
| **fabricated** | Asserts something the reference contradicts, or invents a value. |

**Method, stated plainly**: scored by inspection, one reader comparing each recorded answer against the committed golden reference. No automated judge and no second rater were involved, so the boundary between `correct` and `partial` in particular carries the reader's judgment. Scored on the recorded 2026-08-16 sweep, not re-run since. The verdicts below are reproducible from the recorded run in the sense that the answers are fixed text; they are not reproducible in the sense that a different reader might move a borderline row.

### Headline

- **Of the 23 answerable questions, 11 were correct.** 3 were partial, 8 were refused, and 1 was fabricated.
- Of the 2 unanswerable questions, both were correctly refused, with no chunks retrieved and no invented content. That result stands unchanged.
- The rubric changes the story. Under the old "answered vs refused" count, 15 of 23 looked like a pass and the report claimed zero fabrications. Reading the answers, 4 of those 15 do not hold up: 3 are partial and 1 is a fabrication. **The claim of zero fabrications across the sweep was wrong** and has been removed from this report and from `README.md`, `doc/architecture.md` and `doc/presentation/deck.md`. The corrected figure is 11 of 23 correct, i.e. under half the answerable set.
- The single fabrication is question 9: the model asserted that the filed form does not specify a validity period, where the reference says it does and names one. That is an assertion of absence contradicted by the reference, which the rubric scores as fabricated even though no value was invented.
- Two of the three partials are quality failures visible in the answer text itself, not disagreements about the fact: question 6 emitted the synthesizer's own expected-output instruction as the first line of the answer before stating the (correct) figure, and question 11 produced four repetitive, mutually inconsistent sentences about its single retrieved chunk and concluded it could not determine the comparison. Question 7 answered with the form's generic statutory title instead of the specific purpose the reference gives.

### Results by category (rubric)

| Category | Correct | Partial | Refused | Fabricated | Total |
|---|---|---|---|---|---|
| single-doc-numeric | 4 | 1 | 1 | 0 | 6 |
| single-doc-factual | 7 | 1 | 6 | 1 | 15 |
| multi-doc | 0 | 1 | 1 | 0 | 2 |
| unanswerable (refusal is the correct verdict) | n/a | 0 | 2 | 0 | 2 |
| **answerable total** | **11** | **3** | **8** | **1** | **23** |

Neither multi-document question was answered correctly. That is 0 for 2 on the category the corrective-RAG design is most meant to help with, on a corpus that does contain both halves of both comparisons.

### Results by source document (rubric)

| Document | Correct | Partial | Refused | Fabricated | Total |
|---|---|---|---|---|---|
| Timesheet (French timesheet template) | 4 | 0 | 0 | 0 | 4 |
| Form No. 42 (receipt + filed form + cross-document) | 4 | 3 | 3 | 1 | 11 |
| Invoice (June 2026) | 2 | 0 | 2 | 0 | 4 |
| Payslip (June 2026) | 1 | 0 | 3 | 0 | 4 |

The timesheet is the only document the pipeline handles cleanly, 4 for 4. Refusals concentrate in the payslip (3 of 4) and the invoice (2 of 4). Form No. 42 has the most questions and the widest spread of failure modes: it accounts for all 3 partials and the only fabrication, so its earlier "8 of 11 answered" reading was the most misleading row in the old table.

### Latency distribution (25 questions, agentic mode)

| Stat | Value |
|---|---|
| min | 60.2 s |
| p50 (median) | 125.1 s |
| mean | 132.3 s |
| max | 257.6 s |
| total (sum across all 25) | 3308.1 s (~55 min) |

These latencies were recorded before the Researcher stage stopped being a CrewAI agent (see `doc/architecture.md`). That change removes one LLM completion per retrieval attempt without changing what is retrieved. One question from this set was re-run afterwards against the same corpus and model: question 1, recorded at 120.6 s here, returned the same correct answer in **59.2 s**. That is a single data point, one question, one run, not a re-measured distribution, and the table above has deliberately not been adjusted for it. A full re-sweep is needed before any median is restated.

### Per-question results (rubric)

Answers are described, never quoted, because the corpus is a set of real personal documents. Where a row failed, the note says how it failed without restating the value involved.

| # | Category | Question (short) | Verdict | Chunks | Latency (s) | Note |
|---|---|---|---|---|---|---|
| 1 | single-doc-factual | Form 42: what is it used for | Correct | 4 | 120.6 | |
| 2 | single-doc-numeric | Form 42: e-filing acknowledgement number | Correct | 1 | 60.2 | |
| 3 | single-doc-factual | Form 42: e-filing date | Refused | 0 | 173.3 | |
| 4 | single-doc-factual | Form 42: tax year | Correct | 1 | 106.3 | |
| 5 | single-doc-factual | Form 42: filing type (original/revised) | Correct | 1 | 85.7 | |
| 6 | single-doc-numeric | Form 42: attachment count | **Partial** | 1 | 140.4 | the figure is right, but the answer opens by echoing the synthesizer's own expected-output instruction as if it were content: a visible prompt leak, so not scorable as correct |
| 7 | single-doc-factual | Form 42: purpose of the TRC | **Partial** | 1 | 66.5 | answered with the form's generic statutory title instead of the specific purpose the reference states; right document, wrong level of detail |
| 8 | single-doc-factual | Form 42: applicant nationality | Refused | 0 | 162.9 | |
| 9 | single-doc-factual | Form 42: TRC applicability period | **Fabricated** | 1 | 133.5 | asserted the form does not specify a validity period; the reference says it does and names one |
| 10 | single-doc-factual | Form 42: verification capacity (individual/company) | Refused | 0 | 105.1 | |
| 11 | multi-doc | Form 42: receipt vs. filed form, same ack. number? | **Partial** | 1 | 182.5 | four repetitive sentences about one retrieved chunk, contradicting each other about what that chunk contains, ending in "cannot be determined"; the reference says the two documents do match. Scored partial rather than fabricated because it makes no positive false claim about the documents' contents, but it delivers no usable answer |
| 12 | single-doc-numeric | June 2026 invoice: invoice number | Refused | 0 | 86.2 | |
| 13 | single-doc-numeric | June 2026 invoice: total amount | Correct | 2 | 75.7 | |
| 14 | single-doc-factual | June 2026 invoice: line item billed | Correct | 2 | 134.4 | |
| 15 | multi-doc | Compare the two June 2026 invoice files | Refused | 0 | 125.1 | |
| 16 | single-doc-numeric | Timesheet: total hours | Correct | 1 | 174.4 | |
| 17 | single-doc-numeric | Timesheet: timesheet number | Correct | 1 | 115.3 | the groundedness checker flagged this one as ungrounded even though it matches the reference; a false positive on the checker's side, not an answer defect |
| 18 | single-doc-factual | Timesheet: week covered | Correct | 1 | 108.1 | |
| 19 | single-doc-factual | Timesheet: submitted/approved dates | Correct | 1 | 257.6 | |
| 20 | single-doc-factual | Payslip: designation/job title | Refused | 0 | 162.9 | |
| 21 | single-doc-factual | Payslip: payment mode | Refused | 0 | 158.0 | |
| 22 | single-doc-factual | Payslip: pay cycle date range | Refused | 0 | 190.2 | |
| 23 | single-doc-factual | Payslip: date joined | Correct | 1 | 108.6 | |
| 24 | unanswerable | Company stock price | Refused (correct) | 0 | 70.9 | |
| 25 | unanswerable | Annual bonus amount | Refused (correct) | 0 | 203.7 | |

**A note on the groundedness flag.** It is not a correctness signal and should not be read as one. Of the three answers the checker marked ungrounded (6, 11, 17), one is a correct answer (17), one is a correct figure behind a prompt leak (6), and one is the garbled comparison (11). Meanwhile the one fabrication (9) was marked grounded. On this run the flag and the rubric agree on roughly half the rows they both cover, which is another reason the rubric had to be scored by reading the answers.

### Cross-mode diagnostic: isolating the cause of the refusals

Four of the eight refused questions above were re-run in `simple` mode, which shares the same `HybridRetriever` and index as `agentic` mode but skips the corrective grading/synthesis-check loop. All four were answered correctly. **This is a 4-question spot check, not a rubric run**: `simple` mode has never been scored on the four-way rubric across all 25 questions, so the two modes are not rubric-comparable and no "simple beats agentic on correctness, N to M" claim is available from this report.

| Refused question (agentic) | Simple-mode result | Chunks | Latency (s) |
|---|---|---|---|
| Form 42: e-filing date | Correct | 7 | 27 |
| June 2026 invoice number | Correct | 8 | 25 |
| Payslip: designation | Correct | 10 | 82 |
| Payslip: payment mode | Correct | 11 | 60 |

Since both modes read from the same retrieval index, a question that `simple` mode answers correctly and `agentic` mode refuses cannot be a retrieval problem: the relevant chunks are present and rankable. The difference has to be downstream of retrieval, in the per-chunk relevance grading stage that only `agentic` mode runs. The `qwen2.5:3b` grader is known to be fragile on this exact classification task (see `doc/architecture.md` for the CrewAI-agent-wrapper failure mode that causes it). A previously tried fail-open fallback around the grader, which proceeded to synthesis on the top-ranked chunks whenever the grader rejected all of them, fixed these refusals but was reverted after it caused the synthesizer to fabricate a bonus figure by misattributing an unrelated payslip table row (see Part 2's judge-reliability note for related detail).

### Analysis: what the rubric says, and the concrete next steps

`agentic` mode answers under half the answerable set correctly: 11 of 23. Its dominant failure mode is still refusal (8 of 23), all traced to the grading stage rejecting chunks it should have kept. But refusal is no longer the whole story. Of the 15 responses that were not refusals, 4 are defective: 3 partial and 1 fabrication. So the honest characterisation is not "precision-oriented, trades recall for zero fabrications". It is: high refusal rate, and among the answers it does produce, roughly one in four is wrong or degraded. It does get the two genuinely unanswerable questions right, 2 of 2, which is the one unqualified good result here.

Per-question latency (60 to 258 seconds, mean 132 seconds) is far higher than `simple` mode's, so each of those defective answers also costs about two minutes.

`simple` mode has 9 measured data points (the 5-question RAGAs run in Part 2 and the 4-question cross-mode diagnostic above), and answered all of them correctly, at 25 to 82 seconds in the diagnostic. **It has not been scored on this rubric.** "Simple mode answers essentially everything" is therefore scoped to 9 questions judged loosely, against agentic mode's 25 questions judged strictly, and the two figures must not be put side by side as if they were the same measurement.

Next steps, in priority order:

1. **Run the same 25 questions through `simple` mode and score them on this identical four-way rubric.** This is the missing measurement. Until it exists, the choice of `simple` as the default rests on latency and on 9 loosely-judged questions, not on a like-for-like correctness comparison. Nothing else in this report should be read as ranking the modes on correctness.
2. Run the per-chunk grading stage on a larger model, `qwen2.5:7b`, which is already pulled locally and already used as the RAGAs judge, instead of the `qwen2.5:3b` used for the other agent roles. This 3B model's grading judgment is already known to be fragile to wrapper and prompt framing (see `doc/architecture.md`), which is the most likely explanation for the 8 refusals. Re-run the 25 questions with only that stage swapped, so the comparison isolates one change.
3. Fix the prompt leak behind question 6's partial. The synthesizer emitted its own expected-output instruction into the answer body, which is a formatting defect in the synthesis stage, independent of retrieval or grading quality.
4. Replace inspection scoring with a second rater or a larger-model judge on the same rubric, so the correct/partial boundary stops resting on one reader.

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
