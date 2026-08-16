# DocuMind

DocuMind is a document search platform with an agentic, self-correcting RAG backend. Drop PDFs into a folder, ingest them, and ask questions about them through a normal chat UI (OpenWebUI) or a plain REST API. The backend is a FastAPI service that combines Docling for PDF parsing, LlamaIndex + PGVector for hybrid retrieval, CrewAI for a five-role corrective-RAG crew plus a direct-call relevance grading stage, Ollama for fully local inference, and Arize Phoenix for tracing and prompt management.

It was built as a technical assessment: mandated stack (Docling, PGVector, LlamaIndex, CrewAI, Ollama, Phoenix, RAGAs, OpenWebUI), CPU-only, one-command startup.

**What it does:**

- **Agentic corrective RAG** (opt-in, see Section 3). A CrewAI crew routes, rewrites, retrieves, synthesizes, and runs a groundedness check over its own answer. The relevance-grading step in between runs as a direct LLM call rather than a CrewAI agent, because wrapping it in a CrewAI `Agent`/`Task` was measured to invert the model's judgment: the agent wrapper's injected system message made the small model answer "no" to everything, including a control question with an obvious answer. Two independently bounded correction loops (bad retrieval triggers one rewrite retry; an ungrounded answer triggers one regeneration).
- **Hybrid search.** Every query combines dense vector similarity (PGVector/HNSW cosine) with Postgres full-text search (GIN on `tsvector`), fused by LlamaIndex, so exact identifiers (invoice numbers, acknowledgement numbers) are not lost to pure embedding similarity.
- **Full tracing.** Every HTTP request, agent step, tool call, and LLM completion is exported to Phoenix as an OpenTelemetry span, with prompts, completions, and token counts.
- **PromptOps.** All agent prompts live in git-versioned YAML, sync to Phoenix's prompt hub on startup, and are pulled back at runtime so a prompt edited in the Phoenix UI takes effect without a code change or redeploy.
- **Evaluation.** An offline RAGAs harness (`scripts/evaluate.py`) scores the agentic pipeline against a naive one-shot baseline on a hand-curated golden set, judged by a local model. See `doc/evaluation-report.md` (see Section 7).
- **OpenAI-compatible API.** OpenWebUI talks to DocuMind as if it were a normal OpenAI-compatible chat model; no custom OpenWebUI plugin code.

---

## 1. Architecture at a glance

| Service | Image / tech | Role |
|---|---|---|
| `backend` | FastAPI, Python 3.12, uv | REST APIs, agentic RAG pipeline, ingestion |
| `postgres` | `pgvector/pgvector:pg16` | Vector store (HNSW) + full-text search (GIN) + ingestion ledger |
| `ollama` | `ollama/ollama` | LLM + embedding serving, CPU |
| `phoenix` | `arizephoenix/phoenix` | Tracing, prompt hub, debugging UI |
| `openwebui` | `ghcr.io/open-webui/open-webui` | Chat frontend, talks to the backend as an OpenAI connection |
| `ollama-init` | one-shot | Pulls `qwen2.5:3b`, `nomic-embed-text`, `qwen2.5:7b` on first boot, then exits |

```mermaid
graph LR
    User(("User"))
    OpenWebUI["OpenWebUI\n(chat frontend)"]

    subgraph Backend["backend (FastAPI)"]
        API["/v1/chat/completions\n/api/v1/query"]
        Crew["CrewAI crew\nRouter -> Rewriter -> Researcher\n-> Grader -> Synthesizer -> Checker"]
        Retriever["LlamaIndex hybrid retriever"]
        API --> Crew
        Crew --> Retriever
    end

    Postgres[("Postgres + PGVector\n(vectors, full-text, ledger)")]
    Ollama["Ollama\n(qwen2.5:3b / nomic-embed-text)"]
    Phoenix["Phoenix\n(tracing + prompt hub)"]

    User --> OpenWebUI --> API
    Retriever --> Postgres
    Crew -->|LLM calls| Ollama
    Retriever -->|embeddings| Ollama

    Backend -.->|OTLP spans| Phoenix
    Backend -.->|sync/pull prompts| Phoenix
```

Full HLD/LLD, including the ingestion flow, the agentic pipeline's state diagram with its correction loops, and the data model, is in **[doc/architecture.md](doc/architecture.md)**.

---

## 2. Prerequisites

- Docker Desktop (with Compose v2), running.
- About 10 GB of free disk for images plus the three Ollama models.
- No GPU required. Everything (Docling parsing, embeddings, generation) runs CPU-only; this is a deliberate constraint of the assessment, not an oversight, and the latency notes below are calibrated to it.

---

## 3. Quick start

```bash
# 1. Configure
cp .env.example .env
# edit .env and set DOCUMIND_API_KEY to something other than the default

# 2. Start the full stack
docker compose up -d --build

# 3. Wait for models to finish pulling (first boot only, a few minutes depending on bandwidth)
docker compose logs -f ollama-init
# it exits 0 once qwen2.5:3b, nomic-embed-text and qwen2.5:7b are all present

# 4. Put PDFs into data/documents/ (this folder is gitignored; see the note below), then ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer <your-key>"

# check progress with the job id it returns
curl http://localhost:8000/api/v1/ingest/<job_id> -H "Authorization: Bearer <your-key>"

# 5. Chat
# open http://localhost:3000, it is pre-wired to the backend's OpenAI-compatible API
# pick the "agentic-rag" model and ask a question about your documents
```

The chat frontend advertises a single model id, `agentic-rag`. That id is a fixed label, not a mode selector: which pipeline it actually runs follows `DOCUMIND_PIPELINE_MODE`, which now ships as `simple`. Set it to `agentic` (and restart the backend) if you want the chat UI to drive the corrective crew.

`data/` is gitignored by design: the sample documents used during development are real personal records (a payslip, invoices, a filed tax form), so they are never committed. Put your own PDFs there, or use the ones you were given for the assessment.

**Which mode is the default, and why:** `simple` is the shipped default (`DOCUMIND_PIPELINE_MODE=simple`). Against the real corpus, agentic mode answered 15 of 23 answerable golden-set questions with a median latency of 125 seconds, while simple mode answered correctly every question it was tried against, in 25 to 82 seconds, from the same index. The 8 agentic refusals came from the per-chunk relevance grader rejecting every retrieved chunk, not from a retrieval failure (Section 7). Defaulting to the mode that makes you wait two minutes to more often be told "I couldn't find anything" would make the first thing anyone tries the worst thing the system does. Note the asymmetry in evidence: agentic mode has a full 25-question sweep, simple mode has 9 measured questions, so read "answers essentially everything" as scoped to those 9, per `doc/evaluation-report.md`.

Agentic mode is fully supported and unchanged, just opt-in. Turn it on either way:

- per deployment: `DOCUMIND_PIPELINE_MODE=agentic` in `.env`, then restart the backend. This also changes what the chat UI runs.
- per request: `{"mode": "agentic"}` in the body of `POST /api/v1/query`, which overrides the configured default for that call only.

**Expected latency (CPU, no GPU):** this is the part most likely to look "stuck" if you don't know what to expect.

- **Agentic mode** (`DOCUMIND_PIPELINE_MODE=agentic`, or `"mode": "agentic"` per request; opt-in): a full answer runs the crew's stages (route, rewrite, retrieve, grade, synthesize, check), some of which loop. Measured against the real, fully populated 6-document corpus across all 25 golden-set questions (full distribution in `doc/evaluation-report.md`): **min 60s, median (p50) 125s, mean 132s, max 258s.** An earlier ~29s/~37s figure that circulated during development was measured against an empty index (no documents ingested yet), which short-circuits retrieval and grading and is not representative; the numbers above are the honest, corpus-backed ones and supersede it everywhere in this project's docs. Typical case is 5-8 LLM completions; the per-chunk relevance grader can add up to `retrieval_top_k` (default 6) more single-token completions on top of that if a correction loop fires. This is the pipeline the assessment is built around: the user requirement was explicitly to trade latency for agentic depth (full corrective crew, not a leaner one) rather than optimize for speed.
- **Simple mode** (`DOCUMIND_PIPELINE_MODE=simple`, the default; or `"mode": "simple"` per request on `/api/v1/query`): retrieve once, synthesize once, no grading or self-correction. Measured against the real corpus at 25 to 82 seconds per question, depending on the question (see `doc/evaluation-report.md`). This is both the default path and the "naive RAG" baseline the evaluation report compares the agentic pipeline against.
- The chat UI streams stage-status updates ("Routing...", "Retrieving...", "Grading context...") inside a collapsible "thinking" panel while the agentic pipeline runs, specifically so a long wait doesn't look like a hang.
- None of the above is a timeout misconfiguration: it is real CPU inference time for a 3B generation model doing several sequential completions. If you have a GPU-backed Ollama, point `DOCUMIND_OLLAMA_BASE_URL` at it and latency drops accordingly; nothing else in the design assumes CPU.
- Agentic mode also has a measured recall limitation (it refuses more than it should on some real questions); see the callout at the top of Section 7 before you switch a demo over to it.

---

## 4. Configuration

Every setting is a `DOCUMIND_`-prefixed environment variable (via `pydantic-settings`), documented in `.env.example`. Defaults shown below are what ships if you don't override them.

| Variable | Default | Purpose |
|---|---|---|
| `DOCUMIND_API_KEY` | `documind-dev-key` | Bearer token required on every protected endpoint. OpenWebUI is configured to send this as its OpenAI API key. |
| `DOCUMIND_DATABASE_URL` | `postgresql+psycopg://documind:documind@localhost:5432/documind` | Postgres connection string (vector store + ingestion ledger). |
| `DOCUMIND_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for both generation and embeddings. |
| `DOCUMIND_PHOENIX_BASE_URL` | `http://localhost:6006` | Phoenix server for tracing export and the prompt hub. |
| `DOCUMIND_LLM_MODEL` | `qwen2.5:3b` | Generation model for routing, rewriting, grading, synthesis, and the hallucination check. |
| `DOCUMIND_EMBED_MODEL` | `nomic-embed-text` | Embedding model, 768 dimensions. |
| `DOCUMIND_EMBED_DIM` | `768` | Must match `embed_model`'s output size; used to size the PGVector column. |
| `DOCUMIND_JUDGE_MODEL` | `qwen2.5:7b` | Default judge model for the offline RAGAs evaluation harness (`scripts/evaluate.py`'s `--judge-model` defaults to this setting); override with `--judge-model` for a run that needs a smaller/faster judge. |
| `DOCUMIND_LLM_TIMEOUT_SECONDS` | `180.0` | Timeout for a single LLM completion. |
| `DOCUMIND_PIPELINE_MODE` | `simple` | `simple` (retrieve then synthesize) or `agentic` (full corrective crew). Overridable per request on `/api/v1/query`. See Section 3 for why `simple` is the default. |
| `DOCUMIND_RETRIEVAL_TOP_K` | `6` | Chunks retrieved per query, and the cap on how many chunks the per-chunk grader will grade. Overridable per request via `/api/v1/query`'s `top_k` (1 to 50). |
| `DOCUMIND_MAX_RETRIEVAL_ATTEMPTS` | `2` | Bound on the rewrite-retrieve-grade loop: one attempt plus at most one correction. |
| `DOCUMIND_MAX_GENERATION_ATTEMPTS` | `2` | Bound on the synthesize-check loop: one attempt plus at most one regeneration. |
| `DOCUMIND_REQUEST_BUDGET_SECONDS` | `300.0` | Whole-request wall-clock budget for agentic mode. Stops the pipeline from starting more work once exhausted and returns the best result it already has; cannot interrupt a completion already in flight (that's `LLM_TIMEOUT_SECONDS`'s job). |
| `DOCUMIND_CHAT_STREAM_MAX_WORKERS` | `4` | Size of the dedicated thread pool that runs agentic pipeline calls for the streaming `/v1/chat/completions` endpoint, kept separate from the shared default executor so an aborted stream can only ever starve this one endpoint, not the rest of the app. |
| `DOCUMIND_DATA_DIR` | `data/documents` | Directory scanned for PDFs to ingest. Relative to the process's working directory; see the Development section for what that means when running the backend outside Docker. |
| `DOCUMIND_PROMPTS_DIR` | `prompts` | Directory of YAML prompt templates. Same relative-path caveat as above. |
| `DOCUMIND_CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk when Docling's `HybridChunker` splits a document. |
| `DOCUMIND_VECTOR_TABLE_NAME` | `rag_chunks` | Postgres table name; LlamaIndex prefixes it, so the actual table is `data_rag_chunks`. |

`docker-compose.yml` also uses three non-prefixed variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`) purely to configure the Postgres image and to build `DOCUMIND_DATABASE_URL` for the `backend` service; they are not read by the application itself.

---

## 5. REST API

All endpoints except `/health` require `Authorization: Bearer <DOCUMIND_API_KEY>`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus dependency checks (Postgres, Ollama, Phoenix). No auth. |
| `GET` | `/api/v1/documents` | List every document in the ingestion ledger with status and chunk counts. |
| `POST` | `/api/v1/documents` | Upload one PDF (multipart) into `data/documents/` and ingest it immediately. |
| `GET` | `/api/v1/documents/{doc_id}` | One document's ledger entry. |
| `DELETE` | `/api/v1/documents/{doc_id}` | Remove a document and its chunks from the index. |
| `POST` | `/api/v1/ingest` | Scan `data/documents/` and (re-)ingest everything new or changed; returns a job id, runs in the background. |
| `GET` | `/api/v1/ingest/{job_id}` | Job progress: total/completed/failed document counts. |
| `POST` | `/api/v1/query` | Direct RAG query: answer, citations, retrieved chunks, a groundedness signal (`grounded`, a weak heuristic, not a guarantee, see `doc/api.md`), Phoenix trace id, latency. Accepts an optional per-request `mode` and `top_k`. Used by the evaluation harness and any non-chat consumer. |
| `GET` | `/v1/models` | OpenAI-compatible model listing (advertises `agentic-rag`), for OpenWebUI's dropdown. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion. Runs the same pipeline as `/api/v1/query`; supports `stream: true` (SSE with live stage-status updates inside a `<think>` block). |

Interactive docs: `http://localhost:8000/docs` (Swagger UI, auto-generated from the same schema). Full request/response shapes and error codes: **[doc/api.md](doc/api.md)**. Machine-readable schema: **[doc/openapi.json](doc/openapi.json)**, exported by `scripts/export_openapi.py` and authoritative over any hand-written description if the two ever disagree.

---

## 6. Observability & PromptOps

Open Phoenix at `http://localhost:6006`.

**Tracing.** Every request produces a trace tree: a root HTTP server span (via `opentelemetry-instrumentation-fastapi`), a `CHAIN` span per CrewAI crew kickoff, an `AGENT` span per CrewAI role (Router, Rewriter, Researcher, Synthesizer, Checker; five roles, not six), a `TOOL` span for each `document_search` call, `LlamaIndex` retriever/embedding spans underneath it, and `LLM` `ChatCompletion` spans carrying the actual prompt, completion, model name, and token counts. The relevance grader is a direct LLM call rather than a CrewAI agent, so it produces an `LLM` span like every other completion, but it never produces an `AGENT` span; no "Grader" agent span can appear in a trace. A `correlation.id` attribute ties every span in a request back to its `X-Correlation-ID` log lines. Live traces were confirmed to carry a non-zero trace id end to end.

**Prompts.** The five graded/generative prompts (router, rewriter, grader, synthesizer, groundedness checker) are externalized in `prompts/*.yaml` (git-versioned, the source of truth) and managed through Phoenix as described below. What is *not* externalized: each CrewAI agent's `role`/`goal`/`backstory` strings, the Researcher's task description (it has no YAML template at all, unlike the other five stages), and the one-off `direct_answer` chit-chat prompt, all of which remain as plain strings in `backend/app/agents/stages.py`. On startup, the backend syncs each YAML template into Phoenix's prompt hub, comparing content first so re-running doesn't pile up duplicate versions, then immediately pulls the current version back so that a change made in the Phoenix UI wins over the YAML file for the rest of that process, without a restart. If Phoenix is unreachable, the app falls back to YAML silently. To change one of the five YAML-backed prompts:

- **Quick experiment / one-off tweak:** edit it directly in the Phoenix UI (Prompts tab). Takes effect immediately for new requests.
- **Permanent change:** edit the `.yaml` file in `prompts/`, bump its `version:` field, and commit it. It syncs to Phoenix on the next backend restart.

---

## 7. Evaluation

**Known limitation: agentic mode trades recall for precision.** This is the measurement behind shipping `simple` as the default mode (Section 3). Run against the real, fully populated corpus, agentic mode answered 15 of 23 answerable golden-set questions and correctly refused both of the 2 unanswerable ones, with zero fabricated answers across all 25 questions. The 8 refusals on answerable questions were all traced to the per-chunk relevance grader rejecting every retrieved chunk before synthesis ever ran, not to a retrieval failure. Simple mode, which skips grading entirely, answered every question it was tried against in the same evaluation. See `doc/evaluation-report.md` for the full breakdown; a fail-open fix for the grader (proceeding with the top-ranked chunks when the grader rejected all of them) was tried and reverted, since it fixed those refusals but caused the synthesizer to fabricate a figure on an unanswerable question. Simple mode is therefore the default: use agentic mode to show the corrective-RAG behavior itself.

**Read `grounded` as a signal, not a guarantee.** The groundedness checker is one yes/no completion from the same 3B model, over the whole retrieved context at once. `true` means only that it did not object. It verifies textual presence, not semantic support, so it has passed an answer that attached a real number to the wrong label, because the digits did appear somewhere in the context. `null` means no check ran at all. Full definition of all three values in `doc/api.md`.

`scripts/evaluate.py` runs the hand-curated golden question set (`evaluation/golden_set.json`) through both pipeline modes via `POST /api/v1/query`, then scores each response with RAGAs (faithfulness, answer relevancy, context precision, context recall), judged by a local model. The script's `--judge-model` defaults to `DOCUMIND_JUDGE_MODEL` (`qwen2.5:7b`); the recorded run in `doc/evaluation-report.md` substituted the smaller `qwen2.5:3b` to fit a CPU time budget, and that substitution is disclosed in the report itself.

`ragas` and `langchain-ollama` are in the non-default `eval` uv dependency group, so `--group eval` is required or the script fails with `ModuleNotFoundError`:

```bash
uv run --project backend --group eval python scripts/evaluate.py \
  --api-key <your-key> \
  --modes simple agentic
```

Add `--limit N` or `--indices 1,4,9,...` to bound wall-clock time on a slow CPU judge; the full golden set stays committed regardless of what a given run evaluates. Results and a written report land in `evaluation/runs/<timestamp>/results.json` (gitignored, per-question detail) and `doc/evaluation-report.md` (committed).

**Judge caveat:** the judge is a 7B model running on CPU, not a frontier model. Absolute scores are noisy; treat them directionally. Because both pipeline modes are scored by the same judge, its bias mostly cancels out of the agentic-vs-naive *comparison*, which is the more load-bearing number here than any single absolute score.

---

## 8. Development

**Local backend setup** (outside Docker; useful for running tests or debugging without rebuilding an image):

```bash
cd backend
uv sync --all-groups
```

**Run the unit test suite** (mocked LLM and Postgres, no services required):

```bash
cd backend
uv run pytest -m "not integration"
```

**Run the integration suite too** (requires a real Postgres + Ollama, e.g. from `docker compose up -d postgres ollama`):

```bash
cd backend
RUN_INTEGRATION=1 uv run pytest
```

**Running the backend itself outside Docker:** `Settings.data_dir` (`data/documents`) and `Settings.prompts_dir` (`prompts`) are relative paths, resolved against the process's current working directory. Inside the container this is `/app` and both are mounted there, so it just works. Locally, either start `uvicorn` from the **repo root** (not from `backend/`), or set `DOCUMIND_DATA_DIR`/`DOCUMIND_PROMPTS_DIR` to absolute paths in your `.env`.

**Repository layout:**

```
documind/
├── README.md                  # this file
├── docker-compose.yml         # full stack, one command up
├── .env.example                # every DOCUMIND_* knob documented
├── backend/
│   ├── app/
│   │   ├── api/               # routers: openai_compat, documents, ingest, query, health
│   │   ├── agents/            # CrewAI crew stages, tools, shared LLM handle
│   │   ├── pipelines/          # agentic (corrective) and simple (naive) pipeline orchestration
│   │   ├── ingestion/         # docling parsing, chunking, contextualizer, pipeline
│   │   ├── retrieval/         # LlamaIndex hybrid retriever, PGVector setup
│   │   ├── observability/     # Phoenix tracing setup, YAML+Phoenix prompt manager
│   │   ├── core/              # config, auth, correlation ids, middleware, errors, env guard
│   │   └── db/                # SQLAlchemy models, ingestion ledger repository, session
│   ├── tests/
│   │   ├── unit/               # mocked LLM/Postgres, always run
│   │   └── integration/       # real Postgres/Ollama, gated on RUN_INTEGRATION=1
│   └── pyproject.toml         # uv-managed; dev + eval dependency groups
├── prompts/                   # externalized agent prompts (YAML, versioned)
├── data/documents/            # PDFs dropped here for ingestion (gitignored, .gitkeep only)
├── evaluation/                # golden Q&A set, RAGAs harness output
├── scripts/                   # ingest.py, evaluate.py, export_openapi.py
└── doc/                        # this documentation set
```

---

## 9. Documentation index

- **[doc/architecture.md](doc/architecture.md)**: system context, ingestion flow, the agentic pipeline's state machine and its bounded correction loops, and the Postgres data model, all as Mermaid diagrams reflecting the implemented code.
- **[doc/api.md](doc/api.md)**: every endpoint, its method, path, auth, real request/response shapes, and error codes.
- **[doc/openapi.json](doc/openapi.json)**: the raw OpenAPI 3 schema, exported from the running FastAPI app. Authoritative if it ever disagrees with the hand-written docs.
- **doc/evaluation-report.md**: RAGAs results and the agentic-vs-naive latency/quality comparison, produced by `scripts/evaluate.py`.
- **[doc/presentation/deck.md](doc/presentation/deck.md)**: the Marp slide deck covering the brief, architecture, mandated-tool mapping, ingestion and agentic pipelines, observability, PromptOps, evaluation, and engineering findings; see `doc/presentation/README.md` for how to render it.
