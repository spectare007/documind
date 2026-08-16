# DocuMind — Design Specification

**Date:** 2026-08-15
**Status:** Approved for planning
**Product:** DocuMind — a document search platform with an Agentic RAG backend, integrated with OpenWebUI as the chat frontend.

---

## 1. Problem Understanding

### 1.1 Objective

Build a production-grade document search platform for a technical assessment. Users chat with a knowledge base of provided PDF documents through OpenWebUI. The backend is a custom Agentic RAG service exposing REST APIs, with full observability, externalized prompts, and measured retrieval/answer quality.

### 1.2 Mandated stack

| Concern | Tool |
|---|---|
| Document preprocessing | Docling |
| Vector database | PostgreSQL + PGVector |
| RAG implementation | LlamaIndex |
| Multi-agent implementation | CrewAI |
| LLM provider | Ollama (local) |
| PromptOps, tracing, observability | Arize Phoenix |
| RAG evaluation | RAGAs |
| Frontend | OpenWebUI |

### 1.3 Constraints & decisions (confirmed with user)

- **Documents:** mixed-content PDFs, downloaded from the shared Drive folder into `data/documents/` locally (gitignored).
- **Runtime:** Docker Desktop on Windows, **CPU-only** — small models required. Generation: `qwen2.5:3b`; embeddings: `nomic-embed-text` (768-dim); RAGAs judge: `qwen2.5:7b`.
- **OpenWebUI integration:** backend exposes an **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`); OpenWebUI connects to it as an OpenAI connection — no code inside OpenWebUI.
- **Agentic approach:** **full corrective-RAG crew** (Approach B) — user explicitly chose agentic depth over latency. Latency mitigations: single-token grader outputs, hard-bounded corrective loops (max 1 retry each), config flag to bypass the full crew, honest latency documentation.
- **Evaluation:** fully local RAGAs judge (`qwen2.5:7b` via Ollama); noise caveat documented.
- **Effort budget:** 2–3 days, all deliverables done thoroughly.
- **Repo:** folder/repo named `documind`, default branch `master`, pushed to GitHub.

### 1.4 Non-functional requirements

- Reproducible one-command startup (`docker compose up`) on a CPU-only machine.
- Idempotent, restartable ingestion; per-document failure isolation.
- Every inference call traced; prompts editable without code changes or redeploys.
- Type-safe Python (pydantic models throughout), structured logging with correlation IDs, bearer-token auth on all endpoints.
- Tests: unit (mocked LLM) + integration (dockerized deps) + RAGAs eval as quality gate.

### 1.5 Risks

| Risk | Mitigation |
|---|---|
| CPU latency makes chat demo painful (worst case ~8 LLM calls) | Streaming agent-status updates in chat; bounded loops; `PIPELINE_MODE=simple` bypass flag; latency documented |
| Small local judge gives noisy RAGAs scores | Documented caveat; agentic-vs-naive comparison is relative, so shared judge bias partially cancels |
| Docling struggles on a specific PDF | Per-document `failed` status with error recorded; run continues |
| Framework version incompatibilities (CrewAI/LlamaIndex/Phoenix instrumentation) | Pin all versions in `pyproject.toml`; verify instrumentation early (Day 2 morning) |
| Ollama cold starts / model load latency | `ollama-init` one-shot service pulls models at first boot; retries with backoff on embedding/LLM calls |

---

## 2. System Architecture

### 2.1 Services (one `docker-compose.yml`)

| Service | Image / Tech | Role |
|---|---|---|
| `backend` | FastAPI, Python 3.12, uv | REST APIs, agentic RAG pipeline, ingestion |
| `postgres` | `pgvector/pgvector:pg16` | Vector store + full-text search + ingestion ledger |
| `ollama` | `ollama/ollama` | LLM + embedding serving (CPU) |
| `phoenix` | `arizephoenix/phoenix` | Tracing, prompt hub, debugging UI |
| `openwebui` | `ghcr.io/open-webui/open-webui` | Chat frontend → backend's OpenAI-compatible API |
| `ollama-init` | one-shot | Pulls `qwen2.5:3b`, `nomic-embed-text`, `qwen2.5:7b` on first boot |

### 2.2 The crew (CrewAI)

Sequential process with conditional flow; LlamaIndex owns retrieval, CrewAI owns orchestration.

1. **Router** — classifies query: `rag` vs `direct` (small talk skips the pipeline). One cheap call.
2. **Query Rewriter** — turns user question + chat history into standalone search queries; decomposes multi-part questions.
3. **Retriever** — executes the LlamaIndex **hybrid retrieval tool** (PGVector cosine + Postgres full-text, fused). Tool execution, not an LLM call.
4. **Relevance Grader** — binary verdict per retrieved chunk (single-token output). Insufficient context → **one** corrective loop back to Rewriter with feedback, then proceed regardless.
5. **Synthesizer** — grounded answer strictly from graded context, inline citations `[doc, section]`.
6. **Hallucination Checker** — binary groundedness verdict. Fail → one regeneration with feedback; response ships with groundedness flag either way.

Worst case ~8 LLM calls, typical ~5. `PIPELINE_MODE=simple` env flag runs retrieve→synthesize only (demo/latency escape hatch; also the "naive" baseline for evaluation).

---

## 3. Ingestion Workflow & Data Design

### 3.1 Pipeline

Triggered by `scripts/ingest.py` or `POST /api/v1/ingest`; runs as a background job with status polling.

1. **Discover** — scan `data/documents/` for PDFs; SHA-256 per file. Known hashes skipped (idempotent); changed files re-ingested with old chunks deleted.
2. **Preprocess (Docling)** — PDF → `DoclingDocument`: layout analysis, reading order, table structure, heading hierarchy; OCR fallback for scanned pages.
3. **Chunk** — Docling `HybridChunker`: structure-aware splits (sections; tables kept intact), token-capped ~512 tokens with small overlap to fit the embedding model.
4. **Contextualize** — prepend context header per chunk: `[{document title} > {section path}]`; tables also get nearest caption/heading. Contextual RAG at ingest time, zero query-time cost.
5. **Embed** — `nomic-embed-text` (768-dim) via Ollama, batched, retry with backoff.
6. **Index** — LlamaIndex `PGVectorStore` in **hybrid mode**: HNSW (cosine) on the vector column + GIN on the `tsvector` column, one table.

### 3.2 Postgres schema

- `documents` (ingestion ledger): `id, filename, sha256, status (pending|processing|completed|failed), error, page_count, chunk_count, ingested_at`.
- `data_rag_chunks` (LlamaIndex-managed): `id, text, embedding vector(768), text_search_tsv, metadata jsonb` — metadata carries doc id, title, section path, page numbers (powers citations).

### 3.3 Failure handling

Corrupt/unparseable PDF → document marked `failed` with error recorded; run continues. Status endpoint reports per-document outcomes.

---

## 4. REST API Surface

FastAPI; OpenAPI auto-generated at `/docs`, exported to `doc/openapi.json`. Bearer-token auth via `API_KEY` env var (OpenWebUI passes it as the OpenAI API key). Correlation ID on every request, propagated to logs and Phoenix traces.

### 4.1 OpenAI-compatible (for OpenWebUI)

| Endpoint | Behavior |
|---|---|
| `GET /v1/models` | Advertises model `agentic-rag` for OpenWebUI's dropdown |
| `POST /v1/chat/completions` | Runs the crew. `stream: true` (SSE): agent progress status lines ("Routing…", "Retrieving…", "Grading context…") then token-streamed answer. Non-streaming returns standard completion object. Citations + groundedness flag appended to answer text. |

### 4.2 Native domain API (`/api/v1`)

| Endpoint | Behavior |
|---|---|
| `POST /documents` | Upload PDF (multipart) into knowledge base |
| `GET /documents`, `GET /documents/{id}` | Ingestion ledger: status, chunk counts, errors |
| `DELETE /documents/{id}` | Remove document + chunks from index |
| `POST /ingest` | Trigger (re-)ingestion; returns job id |
| `GET /ingest/{job_id}` | Job progress, per-document status |
| `POST /query` | Direct RAG query: answer, retrieved chunks with scores, citations, groundedness verdict, Phoenix trace ID. Used by RAGAs harness and API consumers. |
| `GET /health` | Liveness + dependency checks (Postgres, Ollama, Phoenix) |

---

## 5. Observability, PromptOps & Evaluation

### 5.1 Tracing (Phoenix)

OpenInference auto-instrumentation for CrewAI, LlamaIndex, and the LLM client → nested traces via OTLP: every inference call, tool call, retrieval, agent step. Custom span attributes: correlation ID, prompt version, corrective-loop count, groundedness verdict.

### 5.2 PromptOps

- All agent prompts in `prompts/*.yaml` (git-versioned; zero prompts in Python code).
- On startup, backend syncs YAML prompts to **Phoenix Prompt hub**; at runtime pulls latest tagged version from Phoenix → prompts editable in Phoenix UI without redeploy.
- Phoenix unreachable → YAML fallback. Each trace records the prompt version used.

### 5.3 Evaluation (RAGAs)

Offline script `scripts/evaluate.py` (not in the serving path):

- **Test set:** ~15–20 hand-curated golden Q&A pairs over the actual documents (JSON), plus adversarial cases (unanswerable, multi-doc questions).
- **Metrics:** faithfulness, answer relevancy, context precision, context recall — judged by local `qwen2.5:7b`.
- **Comparison:** full agentic pipeline vs. naive RAG (`PIPELINE_MODE=simple`) — quantifies what the crew buys.
- **Performance:** p50/p95 end-to-end latency + per-stage breakdown from the same harness.
- **Outputs:** `doc/evaluation-report.md` + datasets uploaded to Phoenix experiments.

### 5.4 Testing

- **Unit:** chunking, contextualization, citation parsing, OpenAI-schema conformance — mocked LLM.
- **Integration:** against dockerized Postgres + Ollama (pytest-marked, skippable).
- **Quality gate:** the RAGAs eval suite.

---

## 6. Repository Structure

```
documind/
├── README.md                  # setup, config, deployment, usage — the front door
├── docker-compose.yml         # full stack, one command up
├── .env.example               # every knob documented (models, keys, ports)
├── backend/
│   ├── app/
│   │   ├── api/               # routers: openai_compat, documents, ingest, query, health
│   │   ├── agents/            # CrewAI crew, agent & task definitions, tools
│   │   ├── ingestion/         # docling pipeline, chunking, contextualizer, embedder
│   │   ├── retrieval/         # LlamaIndex hybrid retriever, PGVector setup
│   │   ├── observability/     # Phoenix/OpenInference setup, prompt sync client
│   │   ├── core/              # config (pydantic-settings), logging, auth, errors
│   │   └── db/                # schema, migrations, ingestion ledger repo
│   ├── tests/                 # unit + integration
│   └── pyproject.toml         # uv-managed
├── prompts/                   # externalized agent prompts (YAML, versioned)
├── data/documents/            # PDFs dropped here (gitignored, .gitkeep)
├── evaluation/                # golden Q&A set, RAGAs harness, baseline configs
├── scripts/                   # ingest.py, evaluate.py, export_openapi.py
└── doc/                       # deliverable docs, referenced from README
    ├── architecture.md        # HLD/LLD with Mermaid diagrams
    ├── api.md + openapi.json  # REST API documentation
    ├── evaluation-report.md   # RAGAs results, agentic vs naive comparison
    ├── design-decisions.md    # ADR-style rationale
    └── presentation/          # deck (Marp markdown → PDF/PPTX export)
```

---

## 7. Build Plan (2–3 days)

- **Day 1** — Scaffold + docker-compose stack up; ingestion pipeline end-to-end (Docling → chunks → PGVector) with ledger + status APIs; hybrid retriever working via `POST /query` (simple mode, no agents yet).
- **Day 2** — CrewAI crew (all six roles) wired to retrieval tool; OpenAI-compatible endpoint with streaming status; OpenWebUI connected and demoable; Phoenix tracing + prompt sync live.
- **Day 3** — RAGAs harness + golden set + eval runs (agentic vs naive); README, architecture doc, diagrams, deck, OpenAPI export; test pass, cleanup, push to GitHub.

Git from the start: `git init` with `master` default branch; meaningful commits per milestone.
