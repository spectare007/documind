# DocuMind REST API

Base URL: `http://localhost:8000` (or the container's mapped port). Every endpoint except `GET /health` requires:

```
Authorization: Bearer <DOCUMIND_API_KEY>
```

A missing or wrong key returns `401 Unauthorized`. This document describes every route by hand with real shapes taken from the app's own pydantic schemas; **`doc/openapi.json`** (exported by `scripts/export_openapi.py` from the running app, regenerate it any time the API changes) is the authoritative machine-readable source if anything here ever drifts out of sync with it. Interactive, always-current docs are also served at `GET /docs` (Swagger UI) while the backend is running.

---

## Health

### `GET /health`

No auth required.

**Response `200`:**

```json
{
  "status": "ok",
  "postgres": true,
  "ollama": true,
  "phoenix": true
}
```

`status` is `"ok"` only if both `postgres` and `ollama` are reachable; `phoenix` is checked but never gates `status` since tracing is best-effort throughout the app. This is a live response captured from the running stack.

---

## Documents (`/api/v1/documents`)

Auth required on every route below.

### `GET /api/v1/documents`

List every document in the ingestion ledger.

**Response `200`** (live capture, six ingested PDFs, IDs shown are real but harmless: they are opaque ledger row ids, not personal data):

```json
[
  {
    "id": "9975243743a74467a64548d67be2a769",
    "filename": "Form No 42_ARN.pdf",
    "status": "completed",
    "error": null,
    "page_count": 1,
    "chunk_count": 3,
    "ingested_at": "2026-08-16T08:07:06.900125Z"
  },
  {
    "id": "aaf36749f9664cda806fcbfc148eb913",
    "filename": "Form No 42_Filed Form.pdf",
    "status": "completed",
    "error": null,
    "page_count": 2,
    "chunk_count": 4,
    "ingested_at": "2026-08-16T08:07:14.615566Z"
  }
]
```

`status` is one of `pending | processing | completed | failed`. A `failed` document carries a non-null `error` (truncated to 2000 characters) and null `page_count`/`chunk_count`/`ingested_at`. There is exactly one entry per filename: re-ingesting an edited file updates its existing entry in place rather than adding a second one (see `POST /api/v1/ingest`).

### `POST /api/v1/documents`

Upload a single PDF (multipart form, field name `file`) and ingest it synchronously (the request blocks until ingestion finishes, unlike `POST /api/v1/ingest`).

**Request:** `multipart/form-data` with one `file` field, filename ending in `.pdf`.

**Response `201`:** a `DocumentOut`, same shape as above, with `status: "completed"` (or `"failed"` if that specific file didn't parse).

Uploading a filename that already exists overwrites the file on disk and re-ingests it into the **existing** ledger row, so the returned `id` is the one that document already had and its previous chunks are replaced rather than duplicated.

**Errors:**
- `400`: filename doesn't end in `.pdf`.
- `401`: missing/invalid bearer token.
- `422`: malformed multipart body (FastAPI's own validation).

### `GET /api/v1/documents/{doc_id}`

**Response `200`:** one `DocumentOut`.

**Errors:**
- `404`: `{"detail": "document not found"}`
- `401`: missing/invalid bearer token.

### `DELETE /api/v1/documents/{doc_id}`

Removes the document's chunks from the vector store and its ledger row.

**Response:** `204 No Content`.

**Errors:**
- `404`: `{"detail": "document not found"}`
- `401`: missing/invalid bearer token.

---

## Ingestion (`/api/v1/ingest`)

### `POST /api/v1/ingest`

Scans `DOCUMIND_DATA_DIR` for PDFs and (re-)ingests anything new or changed, in a background thread. Already-ingested, unchanged files are skipped (idempotent).

A document's identity is its **filename**, and its SHA-256 is only the change signal. So a file whose bytes changed keeps its existing ledger row and `id`: the hash on that row is updated, the previous version's chunks are deleted from the index under that same `id`, and the new chunks are inserted. There is always exactly one row per filename, and a re-ingested file never appears twice in `GET /api/v1/documents`.

**Response `202`:**

```json
{ "job_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" }
```

**Errors:** `401`: missing/invalid bearer token.

### `GET /api/v1/ingest/{job_id}`

**Response `200`**, a `JobOut`:

```json
{
  "id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  "status": "completed",
  "total_documents": 6,
  "completed_documents": 6,
  "failed_documents": 0,
  "started_at": "2026-08-16T08:06:52.100Z",
  "finished_at": "2026-08-16T08:07:58.300Z"
}
```

`status` is `running | completed | failed`. `completed` here means the *job* ran to completion, not that every document succeeded; check `failed_documents` and `GET /api/v1/documents` for per-document outcomes. A skipped (already up to date) document still counts toward `completed_documents`.

**Errors:**
- `404`: `{"detail": "job not found"}`
- `401`: missing/invalid bearer token.

---

## Query (`/api/v1/query`)

### `POST /api/v1/query`

The native, non-chat entry point into the RAG pipeline. This is what `scripts/evaluate.py` drives, and what any programmatic consumer that isn't a chat UI should use.

**Request:**

```json
{
  "question": "What is the e-Filing Acknowledgement Number shown on the Form No. 42 acknowledgement receipt?",
  "mode": "agentic",
  "top_k": null
}
```

- `question` (required, non-blank after trimming; a blank/whitespace-only value is a `422`).
- `mode` (optional): `"agentic"` or `"simple"`. Omit to use the server's configured `DOCUMIND_PIPELINE_MODE`, which defaults to `simple`.
- `top_k` (optional, integer 1 to 50): how many chunks each search retrieves for this request, overriding `DOCUMIND_RETRIEVAL_TOP_K`. It applies in both modes, and in agentic mode it also caps how many chunks the per-chunk relevance grader grades, so a larger value cannot leave the extra chunks silently retrieved and then discarded. Omit or send `null` for the configured default. Values outside 1 to 50 are a `422`; the upper bound exists because each additional chunk costs a further grader completion on CPU.

**Response `200`** (representative shape; the identifier value below is a synthetic placeholder, not a real acknowledgement number, illustrating the answer format returned for this question with 3 citations and 7 chunks retrieved in ~25s):

```json
{
  "answer": "The e-Filing Acknowledgement Number shown on the Form No. 42 acknowledgement receipt is <REDACTED-ACK-NUMBER>.",
  "citations": [
    {
      "title": "Form No 42_ARN",
      "section_path": "Acknowledgement Receipt of Income Tax Forms",
      "pages": [1]
    },
    {
      "title": "Form No 42_Filed Form",
      "section_path": "Form No. 42",
      "pages": [1, 2]
    }
  ],
  "chunks": [
    {
      "text": "[Form No 42_ARN > Acknowledgement Receipt of Income Tax Forms]\n\n...",
      "score": 0.81,
      "doc_id": "9975243743a74467a64548d67be2a769",
      "title": "Form No 42_ARN",
      "section_path": "Acknowledgement Receipt of Income Tax Forms",
      "pages": [1]
    }
  ],
  "grounded": true,
  "mode": "agentic",
  "trace_id": "4b2f1a0c9e7d4a6b8f3c2e1d0a9b8c7d",
  "latency_ms": 25100
}
```

- `grounded` is a **weak groundedness signal, not a hallucination guarantee**. It is one yes/no completion from the same local 3B model, asked once over the whole retrieved context, and it verifies textual presence rather than semantic support: it has been observed to pass an answer that attributed a real number to the wrong label, because those digits did appear somewhere in the context. Never present `true` to an end user as proof an answer is correct; treat the field as telemetry about whether a cheap check objected. The field keeps its original name for backwards compatibility. Exactly what produces each value:

  | Value | What it means |
  |---|---|
  | `true` | The checker ran and did not reply with an explicit "no". Note the parser is fail-open: any reply that does not start with "no", including an off-format or empty one, counts as `true`. |
  | `false` | The checker ran and replied "no" on every generation attempt (up to `DOCUMIND_MAX_GENERATION_ATTEMPTS`). The answer is still returned, flagged rather than blocked, and the chat endpoint appends a visible warning to its source list. |
  | `null` | No check ran. Three cases: a direct (non-retrieval) reply, no answer was produced at all (nothing retrieved, budget exhausted, or an unexpected stage failure), or the checker itself raised, in which case the answer is kept but is explicitly reported as unchecked rather than as `true`. |

  Simple mode never runs the checker, so it always returns `null`.
- `trace_id` is the OpenTelemetry trace id (hex, 32 chars) for this request's spans in Phoenix; empty string only if tracing failed to initialize.
- `chunks[].text` includes the ingest-time context header (`[title > section path]`), since that is exactly what the LLM saw.

**Errors:**
- `422`: blank `question`, or an invalid `mode` value.
- `401`: missing/invalid bearer token.
- `503`: the upstream Ollama is unreachable or timed out (`{"detail": "The upstream LLM service (Ollama) is unavailable or timed out. Please retry."}`). This is an explicit, designed response, not a generic crash: both pipelines' LLM-unavailable errors are caught specifically so a slow/down model doesn't surface as an opaque `500`.

---

## OpenAI-compatible chat (`/v1`)

This is what OpenWebUI talks to; no DocuMind-specific client code is required, any OpenAI-compatible client works.

### `GET /v1/models`

**Response `200`:**

```json
{
  "object": "list",
  "data": [{ "id": "agentic-rag", "object": "model", "owned_by": "documind" }]
}
```

Live capture from the running backend.

**Errors:** `401`: missing/invalid bearer token.

### `POST /v1/chat/completions`

**Request:**

```json
{
  "model": "agentic-rag",
  "messages": [
    { "role": "user", "content": "What is the acknowledgement number on the filed form?" }
  ],
  "stream": true
}
```

Only the last `user` message is treated as the current question; earlier `user`/`assistant` turns become chat history (with any `<think>...</think>` block stripped back out of prior assistant turns before the pipeline sees them). `system` messages are dropped; the pipeline has no system-prompt concept.

**Non-streaming response `200`** (`stream: false` or omitted), a standard OpenAI chat completion object:

```json
{
  "id": "chatcmpl-3f9a7c1e2b4d4a8f9c0e1d2a3b4c5d6e",
  "object": "chat.completion",
  "created": 1755331200,
  "model": "agentic-rag",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "The e-Filing Acknowledgement Number is <REDACTED-ACK-NUMBER>.\n\n**Sources:** Form No 42_ARN > Acknowledgement Receipt of Income Tax Forms (p. 1) · Form No 42_Filed Form > Form No. 42 (p. 1, 2)"
      }
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

`usage` is always zeroed; the app does not currently track token counts at this layer (Phoenix's LLM spans do carry real token counts, per-completion, if you need that number).

**Streaming response `200`** (`stream: true`): standard `text/event-stream`, `chat.completion.chunk` frames, terminated by `data: [DONE]`. The assistant's `content` opens with a `<think>` block; each pipeline stage boundary streams one line into it (`Routing query…`, `Rewriting query…`, `Searching documents…`, `Grading context…`, `Synthesizing answer…`, `Verifying groundedness…`), then the block closes (`</think>`) and the real answer plus a `**Sources:**` line streams in 48-character chunks. OpenWebUI renders the `<think>` span as a collapsible "thinking" panel, which is the only reason the UI shows any progress at all during a 30-60 second CPU answer, otherwise a client would see nothing until the very end.

**Errors:**
- `422`: no user message found in `messages` (e.g. only a `system` message, or an empty array; note `messages` requires at least one item at the schema level too).
- `401`: missing/invalid bearer token.
- `503` (non-streaming path only): upstream Ollama unreachable/timed out, same structured detail as `/api/v1/query`.
- On the **streaming** path, an upstream failure cannot become a `503` (the `200` status and the opening `<think>` chunk are already sent by the time it could occur); instead the stream emits a visible warning inside the `<think>` block and then ends cleanly with a normal `finish_reason: "stop"` and `[DONE]`, so no OpenAI-compatible client is left hanging on a malformed stream.

---

## Error shape reference

Most errors are FastAPI's default `{"detail": "..."}` for a plain `HTTPException`, or the standard pydantic validation shape for a `422`:

```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "Value error, question must not be blank",
      "type": "value_error"
    }
  ]
}
```

| Code | Meaning | Where it appears |
|---|---|---|
| `400` | Bad request (business-rule rejection, not a schema error) | Non-PDF upload to `POST /api/v1/documents` |
| `401` | Missing or wrong bearer token | Every route except `GET /health` |
| `404` | No row with that id | `GET/DELETE /api/v1/documents/{doc_id}`, `GET /api/v1/ingest/{job_id}` |
| `422` | Request body fails schema/field validation | Blank `question`, bad `mode`, `top_k` outside 1 to 50, malformed multipart, no user message in `messages` |
| `503` | Upstream LLM (Ollama) unreachable or timed out | `/api/v1/query`, non-streaming `/v1/chat/completions` |
