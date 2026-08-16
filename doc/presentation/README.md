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

## Screenshots (optional)

No screenshots are included in this submission; none were fabricated. The deck describes, in text, what each would show instead of embedding a broken image link. If you want to add them, drop these two files in `doc/presentation/img/` with these exact names and the deck's captions will still apply:

- **`img/chat.png`** (an OpenWebUI chat, `http://localhost:3000`, showing a question answered by the `agentic-rag` model, with the collapsible think-block **expanded** so the stage-status lines are visible: `Routing query…`, `Searching documents…`, `Grading context…`, etc., and the final cited answer visible below it).
- **`img/trace.png`** (a Phoenix trace tree, `http://localhost:6006`, for one agentic query: the `CHAIN` crew-kickoff spans, the `AGENT` span per CrewAI role, the `TOOL` `document_search` span, the LlamaIndex retriever/embedding spans underneath it, and at least one `LLM` `ChatCompletion` span with its prompt, completion, and token counts visible).

## Where the numbers come from

Every number cited in the deck (chunk counts, latencies, span/token counts, test counts) was measured against the live stack during development. The evaluation slide quotes the final, full 25-question sweep against the real corpus from `doc/evaluation-report.md`, not the RAGAs metric run over a 5-question subset that was recorded earlier and is kept in that report for its metric detail only (see that report's Part 2 warning).
