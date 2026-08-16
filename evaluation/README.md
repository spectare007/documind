# Evaluation golden set

`golden_set.json` is the 25-question golden set used by the RAGAs evaluation
harness (see `doc/evaluation-report.md`). The knowledge base it targets is a
small real-world corpus: a payslip, two invoices, a French timesheet, and two
filed income-tax Form No. 42 documents. Those source PDFs belong to the
repository owner and are intentionally **not** committed (`data/` is
gitignored).

## Why the reference answers are redacted

Several `reference` values in this file used to quote real personal and
financial details straight out of those documents: a statutory e-filing
acknowledgement number, an invoice number and total, a billing line item, a
timesheet number and hours, a job designation, a payment mode, a pay cycle
range, a "date joined", a declared nationality, and several filing dates.
That is real personal data belonging to a named individual and it should
never have been committed in plain text, even inside an evaluation fixture.

Every such value has been replaced with an obviously synthetic placeholder in
the form `<REDACTED-SOMETHING>` (for example `<REDACTED-ACK-NUMBER>`,
`<REDACTED-INVOICE-NUMBER>`, `<REDACTED-DATE-JOINED>`). The surrounding
sentence, the question text, the `category`, and the intent of every
question are unchanged, and no question was removed. The file still has 25
entries.

## Using this file against the real documents

The questions themselves remain fully valid: if you have the original
documents in `data/documents/`, run the ingestion pipeline and the evaluation
harness as usual, and grade answers against the real corpus. To restore
concrete reference strings for your own run:

1. Open the real source document referenced by each question (the question
   text names which document it targets, e.g. "the June 2026 payslip" or
   "the French timesheet template").
2. Copy the real value that document contains (the acknowledgement number,
   invoice number, date, etc.) in place of the matching `<REDACTED-...>`
   placeholder in that question's `reference` string.
3. Leave every other field (`question`, `category`) untouched.

Do not commit a version of this file with real values filled back in.
