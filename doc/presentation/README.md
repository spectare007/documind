# Presentation

`deck.md` is a [Marp](https://marp.app/) deck, 14 slides, covering the assessment brief, architecture, the mandated-tool mapping, the ingestion and agentic-RAG pipelines, observability, PromptOps, evaluation methodology, and the engineering findings from the build.

## Rendering

Render to PDF:

```bash
npx @marp-team/marp-cli deck.md -o deck.pdf
```

Render to a self-contained HTML file instead:

```bash
npx @marp-team/marp-cli deck.md -o deck.html
```

Or skip rendering entirely and present `deck.md` directly: any Marp-aware editor (the VS Code "Marp for VS Code" extension, for example) shows it as slides in a preview pane, and it reads fine as plain Markdown even without one.

## Screenshots

Two images are referenced by the deck and are not yet captured. Drop them in `doc/presentation/img/` with these exact names:

- **`img/chat.png`** — an OpenWebUI chat (`http://localhost:3000`) showing a question answered by the `agentic-rag` model, with the collapsible think-block **expanded** so the stage-status lines are visible (`Routing query…`, `Searching documents…`, `Grading context…`, etc.), and the final cited answer visible below it.
- **`img/trace.png`** — a Phoenix trace tree (`http://localhost:6006`) for one agentic query: the `CHAIN` crew-kickoff spans, the `AGENT` span per role, the `TOOL` `document_search` span, the LlamaIndex retriever/embedding spans underneath it, and at least one `LLM` `ChatCompletion` span with its prompt, completion, and token counts visible.

The deck degrades gracefully if these files are absent — Marp just renders a broken-image icon in their place, and the surrounding text still describes what each screenshot would show, so the narrative doesn't depend on the images being present.

## Where the numbers come from

Every number cited in the deck (chunk counts, latencies, span/token counts, test counts) was measured against the live stack and is recorded in `.superpowers/sdd/2026-08-16-documind-implementation/progress.md`. The evaluation slide intentionally does not quote RAGAs scores: at the time this deck was written, `doc/evaluation-report.md` had not yet been produced (the evaluation run was still executing). Once it exists, update that slide with the real numbers from that file rather than editing this deck's methodology framing.
