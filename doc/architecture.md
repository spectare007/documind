# DocuMind Architecture

This document describes what was actually built: HLD/LLD for the ingestion pipeline, the agentic RAG pipeline, and the Postgres data model. Every diagram below reflects the code under `backend/app/` as it exists on `feat/implementation`, not the original proposal; where the two differ, a note says so.

---

## 1. System context

```mermaid
graph TB
    User(("User"))

    subgraph Frontend
        OpenWebUI["OpenWebUI\n:3000\nOpenAI-compatible client"]
    end

    subgraph Backend["backend :8000 (FastAPI, Python 3.12)"]
        OpenAICompat["/v1/models\n/v1/chat/completions"]
        NativeAPI["/api/v1/documents\n/api/v1/ingest\n/api/v1/query"]
        Health["/health"]
        Pipelines["AgenticPipeline / SimplePipeline"]
        Ingestion["IngestionPipeline"]
        OpenAICompat --> Pipelines
        NativeAPI --> Pipelines
        NativeAPI --> Ingestion
    end

    Postgres[("postgres :5432\npgvector/pgvector:pg16\ndata_rag_chunks, documents,\ningest_jobs")]
    Ollama["ollama :11434\nqwen2.5:3b (generation)\nnomic-embed-text (embeddings)\nqwen2.5:7b (RAGAs judge only)"]
    Phoenix["phoenix :6006 / :4317\ntracing (OTLP) + prompt hub"]

    User --> OpenWebUI --> OpenAICompat
    User -.->|curl / scripts| NativeAPI
    Pipelines --> Ollama
    Ingestion -->|Docling parse, then embed| Ollama
    Pipelines --> Postgres
    Ingestion --> Postgres
    Backend -.->|OTLP spans:\nHTTP, crew, agent, tool,\nretriever, LLM| Phoenix
    Backend -.->|sync + pull\nYAML prompts| Phoenix
    Health -.-> Postgres
    Health -.-> Ollama
    Health -.-> Phoenix
```

`ollama-init` (not shown: it is a one-shot container, not a running service) pulls all three models on first boot and exits 0; every other service depends on `ollama` being healthy, not on `ollama-init` having finished, so the backend can start before models are ready and simply fail individual LLM calls until they are.

---

## 2. Ingestion flow

Triggered by `scripts/ingest.py`, `POST /api/v1/ingest` (background job, polled via `GET /api/v1/ingest/{job_id}`), or `POST /api/v1/documents` (single-file, synchronous). All three call into the same `app.ingestion.pipeline.IngestionPipeline`.

```mermaid
flowchart LR
    Discover["Discover\nglob data/documents/*.pdf"]
    Sha["SHA-256\nper file"]
    Known{"Row for this filename,\nsame hash,\nstatus=completed?"}
    Skip["Skip\n(counts as completed\nfor job progress)"]
    Docling["Docling\nDocumentConverter.convert()\nlayout, reading order, tables"]
    Chunk["HybridChunker\nstructure-aware, token-capped\n(max_tokens, default 512)"]
    Ctx["Contextualize\nprepend [title > section path]\n(+ '| table' marker)"]
    Embed["Embed\nnomic-embed-text via Ollama\nget_text_embedding_batch"]
    Index["Index\nPGVectorStore.add()\n(delete old chunks for this\nstable doc_id first, then insert)"]
    Ledger[("documents ledger\npending -> processing ->\ncompleted | failed")]

    Discover --> Sha --> Known
    Known -- yes --> Skip
    Known -- no --> Docling
    Docling --> Chunk --> Ctx --> Embed --> Index

    Sha -. "find row by filename\n(create if absent),\nupdate sha256,\nmark_processing" .-> Ledger
    Index -. "mark_completed\n(page_count, chunk_count)" .-> Ledger
    Docling -. "on exception:\nmark_failed(error),\nrun continues to next doc" .-> Ledger
```

**Concurrency and safety notes actually implemented:**

- `DocumentConverter` (Docling) is not documented thread-safe, so a single process-wide converter is shared behind an `RLock`; conversion is fully serialized across concurrent ingestion jobs. Correctness over throughput, since ingestion is not the hot path.
- **A document's identity is its filename; the SHA-256 is only a change signal.** There is exactly one ledger row per filename, so a document's `doc_id` is stable for the life of the corpus, which is what makes it usable as the vector store's `ref_doc_id` across re-ingests. Re-ingesting an already-completed document whose hash is unchanged is skipped entirely (idempotency). A file whose bytes changed is *not* treated as a new document: the existing row is reused, its `sha256` is updated to the new value, the previous version's chunks are deleted under that same `doc_id` (`store.delete(ref_doc_id=doc_id)`), and the new chunks are inserted. This was a real defect, not a hypothetical: keying identity on the hash meant a changed file missed the lookup and got a *new* row and a *new* `doc_id`, so the delete targeted an id that had no chunks yet. It was a guaranteed no-op, the superseded version's chunks stayed in the index permanently, retrieval mixed stale and current text, and the same filename appeared twice in `GET /api/v1/documents`.
- Because the hash is a change signal rather than an identity, it is deliberately **not** unique: two differently named files may legitimately hold identical bytes and each deserves its own row and its own citations.
- Per-document failure isolation: `IngestionPipeline.run()` wraps each document in its own `try/except`; one corrupt PDF is recorded as `failed` with its error message and the job continues to the next document rather than aborting the whole batch.
- A skipped (already-ingested) document still counts toward the job's `completed_documents`, so a job's progress can reach 100% even on a run where every document was already up to date.
- **TorchDynamo is disabled** (`TORCHDYNAMO_DISABLE=1` in `backend/Dockerfile`). Docling's model path invokes `torch.compile`, which needs a C++ compiler the slim base image doesn't have; since this all runs CPU-only anyway, compilation buys no speedup, so it's turned off rather than adding a toolchain to the image.

---

## 3. Agentic pipeline (corrective RAG)

`app.pipelines.agentic.AgenticPipeline` owns this state machine; `app.agents.stages.CrewStages` owns turning each state into either a CrewAI crew kickoff (a single agent, single task, run to completion) or, for one stage, a direct LLM call, and parsing its output. Five CrewAI agent roles: Router, Query Rewriter, Researcher, Answer Synthesizer, Groundedness Checker. The sixth stage, Relevance Grading, is *not* a CrewAI agent: `CrewStages.grade()` calls `self.llm.call(...)` directly, with no `Agent`, `Task`, or `Crew` involved, because wrapping the per-chunk yes/no verdict in a CrewAI `Agent` was measured to invert the model's judgment: the agent wrapper injects its own system message, and that system message alone flipped the small model to answering "no" regardless of content, confirmed with a control question that has an obvious answer ("Is grass green?", answered "no" with the system message present and "yes" with it removed). Removing the wrapper for this one stage and calling the model directly restored genuine per-chunk discrimination.

```mermaid
stateDiagram-v2
    [*] --> Route
    Route --> DirectAnswer: classified "direct"\n(small talk / about the assistant)
    DirectAnswer --> [*]

    Route --> Rewrite: classified "rag"

    state "Retrieval loop (max_retrieval_attempts = 2)" as RetrievalLoop {
        Rewrite --> Research: standalone search queries
        Research --> Grade: chunks found
        Research --> NoMatch: no chunks found
        Grade --> HaveContext: at least one chunk\ngraded relevant
        Grade --> NoneRelevant: every graded chunk\nrejected ("no")
    }

    NoMatch --> Rewrite: attempt < 2\n(feedback: broaden terms)
    NoneRelevant --> Rewrite: attempt < 2\n(feedback: be more specific)
    NoMatch --> NoContextAnswer: attempts exhausted\nor wall-clock budget hit
    NoneRelevant --> NoContextAnswer: attempts exhausted\nor wall-clock budget hit
    NoContextAnswer --> [*]

    HaveContext --> Synthesize

    state "Generation loop (max_generation_attempts = 2)" as GenerationLoop {
        Synthesize --> Check: grounded answer,\ninline [title, section] citations
        Check --> Grounded: verdict = yes
        Check --> Ungrounded: verdict = no
        Check --> Unchecked: checker raised
    }

    Ungrounded --> Synthesize: attempt < 2\n(feedback: cite only\ngiven context)
    Ungrounded --> ShipFlagged: attempts exhausted\nor wall-clock budget hit\n(ship answer, grounded=false)
    Grounded --> ShipAnswer
    Unchecked --> ShipUnchecked: ship answer, grounded=null\n(no retry: a crashed verifier\nis not a "no" verdict)
    ShipFlagged --> [*]
    ShipAnswer --> [*]
    ShipUnchecked --> [*]
```

**The three independent bounds** (from `AgenticPipeline`'s module docstring, all load-bearing, none of them redundant with the others):

1. **Attempt bounds.** `max_retrieval_attempts` and `max_generation_attempts` (both default 2): one attempt plus at most one correction, each. This bounds *how many times* a loop can run, not how long it takes.
2. **Wall-clock request budget** (`request_budget_seconds`, default 300s). `llm_timeout_seconds` only bounds a single completion; one request can make roughly a dozen of them in the worst case (router, rewrite/retrieve retry, up to `retrieval_top_k` per-chunk grader calls, synthesis/verification retry), so a degraded Ollama could otherwise hold a worker for a very long time. Checked at stage boundaries only: it stops the pipeline from *starting more work* and returns the best result already in hand. It deliberately never skips the first attempt of either loop (a response that tried nothing is worse than a slow one), and it cannot interrupt a completion already in flight, that is `llm_timeout_seconds`'s job.
3. **Exception containment.** Any unexpected stage failure returns a friendly `PipelineResult` (`grounded=None`) instead of propagating as a bare 500. The one deliberate exception: `LLM_UNAVAILABLE_ERRORS` (an unreachable or timed-out Ollama) is re-raised so `app.api.query` and the OpenAI-compatible endpoint can turn it into a structured `503`, since upstream unavailability is an expected condition with its own designed response, not an unexpected bug. The groundedness checker additionally fails open on its own crash, so a flaky verifier can never discard an answer already produced; that path reports `grounded=null` (no check ran), never `true`, so an unchecked answer stays distinguishable from a checked one.

**What `grounded` is worth.** It is a weak signal and the API documents it as one (`doc/api.md`). The checker is a single yes/no completion from the same 3B model over the whole retrieved context, and it verifies textual presence rather than semantic support: it has passed an answer that attributed a real number to the wrong label, because those digits appeared somewhere in the context. `parse_verdict` is also fail-open, so anything that does not start with "no" counts as `true`. `true` therefore means "a small local model did not object", which is telemetry, not a hallucination guarantee, and it is deliberately not presented as one anywhere user-facing.

**Relevance grading is one binary yes/no call per chunk**, not a single call returning a JSON array of relevant indices, and this was a mid-implementation correction, not the original plan: an earlier version asked the grader for a single JSON array of relevant chunk indices across all chunks at once, but the deployed 3B model returned an empty array for every input, relevant or not, once CrewAI's own task boilerplate was appended, which made the pipeline refuse to answer even when the corpus had the answer. It is bounded to the top `retrieval_top_k` chunks by score, so a broad retrieval cannot turn into an unbounded number of grading calls. Every parser in `app.agents.stages` (route, chunk indices, queries, groundedness verdict) fails toward *keeping more*, not less, on an unreadable model reply: an off-format grader keeps the chunk, an off-format router retrieves, an off-format hallucination check assumes grounded. Only an explicit, well-formed rejection (`no`, or an explicit empty array) is trusted as a real negative judgment.

**Simple mode** (`DOCUMIND_PIPELINE_MODE=simple`) is a separate, much shorter pipeline (`app.pipelines.simple.SimplePipeline`): retrieve once, synthesize once, no grading, no rewriting, no correction loop. It is the RAGAs "naive RAG" baseline the agentic pipeline is compared against, and it is also **the shipped default**, because the comparison went against the agentic pipeline on the thing users notice first: agentic mode answered 15 of 23 answerable golden-set questions at a median of 125s, simple mode answered every question it was tried against in 25 to 82s from the same index. Agentic mode is unchanged and fully supported, selected per deployment (`DOCUMIND_PIPELINE_MODE=agentic`) or per request (`"mode": "agentic"` on `POST /api/v1/query`).

**Per-request `top_k`.** `POST /api/v1/query` accepts an optional `top_k` (1 to 50) that overrides `retrieval_top_k` for that call in both modes. In agentic mode it is threaded into both of the researcher's retrieval paths (the `document_search` tool and the direct fallback, so the chunk count does not depend on whether the agent remembered to call its tool) and into the grader's cap, so a larger value cannot leave the extra chunks retrieved and then silently ungraded.

---

## 4. Data model (Postgres)

Three tables, all created by `Base.metadata.create_all()` at startup (no separate migration tool; acceptable for this project's scope). Schema below is read directly from the running database, not from the ORM definitions, so it reflects exactly what LlamaIndex's `PGVectorStore` actually created for the chunk table.

```mermaid
erDiagram
    documents {
        string id PK "uuid4 hex, stable across re-ingests"
        string filename UK "document identity"
        string sha256 "change signal, not unique;\nupdated in place when the file changes"
        string status "pending|processing|completed|failed"
        text error "nullable, truncated to 2000 chars"
        int page_count "nullable"
        int chunk_count "nullable"
        timestamptz ingested_at "nullable"
    }
    ingest_jobs {
        string id PK "uuid4 hex"
        string status "running|completed|failed"
        int total_documents
        int completed_documents
        int failed_documents
        timestamptz started_at
        timestamptz finished_at "nullable"
    }
    data_rag_chunks {
        bigint id PK
        varchar text "contextualized chunk text"
        json metadata_ "doc_id, title, filename,\nsection_path, pages, is_table,\nref_doc_id, node/document ids"
        varchar node_id
        vector embedding "768-dim, HNSW cosine index"
        tsvector text_search_tsv "GIN index, english config"
    }
    documents ||--o{ data_rag_chunks : "doc_id in metadata_"
    ingest_jobs ||--o{ documents : "processed during a run\n(no FK; job/document\nlinked only by timing)"
```

Indexes actually present on `data_rag_chunks` (confirmed against the live instance):

- `data_rag_chunks_pkey`: btree, primary key on `id`.
- `data_rag_chunks_embedding_idx`: HNSW on `embedding`, `vector_cosine_ops`, `m=16`, `ef_construction=64` (search-time `ef_search=40`, set by the app at query time).
- `rag_chunks_idx`: GIN on `text_search_tsv`, this is what makes the lexical half of hybrid search fast.
- `rag_chunks_idx_1`: btree on `metadata_ ->> 'ref_doc_id'`, used by `store.delete(ref_doc_id=...)` during re-ingestion.

One caveat that follows from having no migration tool: `Base.metadata.create_all()` only ever creates missing tables, it never alters an existing one. The `documents` constraints described above (unique on `filename`, non-unique on `sha256`) therefore apply to a freshly created table. A database created before that change keeps its old constraints until the table is recreated, so a deployment carrying old data should drop and re-ingest. Application behaviour does not depend on the constraints: one row per filename is enforced in `IngestionPipeline` by looking the row up by filename before creating one, and the constraints exist to make that invariant explicit at the schema level.

`data_rag_chunks` is LlamaIndex's own table, named by prefixing `Settings.vector_table_name` (`rag_chunks`) with `data_`; the app's own code only ever refers to it as `rag_chunks` via config, never hardcodes the `data_` prefix. `documents` and `ingest_jobs` are hand-rolled SQLAlchemy models (`app/db/models.py`) that exist purely as an ingestion ledger; there is no foreign key between `ingest_jobs` and `documents` because a job's identity is a batch run, not an owner of specific document rows (a document can be created or updated by more than one job over its lifetime, e.g. re-ingestion).

---

## 5. Key design properties

- **Idempotency.** Ingestion is keyed on filename, with the SHA-256 of the file's bytes as the change signal: an unchanged file is skipped, and a changed one reuses its existing row and `doc_id`, updates the stored hash, and has the previous version's chunks deleted before the new ones are added. Running `POST /api/v1/ingest` twice in a row against an unchanged `data/documents/` is a safe no-op, and re-running it after editing a file leaves exactly one version of that document in the index.
- **Per-document failure isolation.** One Docling failure marks that document `failed` with its error text and the batch continues; it does not abort the run or affect any other document's status.
- **Bounded corrective loops.** Both correction loops in the agentic pipeline have a hard attempt cap (default 2 each: one try, one correction), independent of a separate wall-clock budget (default 300s) that stops the pipeline from starting new work once exhausted, regardless of how many attempts remain.
- **Prompt versioning.** Every prompt is YAML, git-versioned, with an explicit `version:` field; Phoenix's prompt hub is a synced, editable mirror (content-hash deduped so repeated syncs don't create redundant versions), not the source of truth, so the system degrades gracefully to YAML if Phoenix is unreachable.
- **Trace propagation.** A `correlation.id` generated (or forwarded from `X-Correlation-ID`) per HTTP request is attached to the active OpenTelemetry span and echoed in logs and the response header; FastAPI/ASGI instrumentation is applied per-app-instance (not gated behind a process-wide flag) specifically so every `create_app()` call, including in tests, gets a real root span rather than a silent no-op one.
