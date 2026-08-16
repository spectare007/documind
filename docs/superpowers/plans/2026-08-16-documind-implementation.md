# DocuMind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DocuMind — an agentic corrective-RAG document search backend (FastAPI) over PDF documents, integrated with OpenWebUI, fully traced in Phoenix, evaluated with RAGAs.

**Architecture:** Docling parses PDFs into structure-aware chunks stored in PGVector via LlamaIndex (hybrid vector + full-text search). A CrewAI corrective-RAG pipeline (router → rewriter → researcher → grader → synthesizer → hallucination checker, bounded loops) answers questions behind an OpenAI-compatible API that OpenWebUI consumes. Prompts live in YAML, synced to Phoenix; every inference call is traced via OpenInference; RAGAs compares agentic vs. naive pipelines with a local Ollama judge.

**Tech Stack:** Python 3.12, uv, FastAPI, Docling, LlamaIndex, PGVector (Postgres 16), CrewAI, Ollama (`qwen2.5:3b`, `nomic-embed-text`, `qwen2.5:7b`), Arize Phoenix, RAGAs, OpenWebUI, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-15-documind-design.md`

## Global Constraints

- Repo root is `documind/`; default branch `master`; run all backend commands from `documind/backend/` with `uv run ...` unless stated otherwise.
- `data/` is gitignored — never commit PDFs or anything under `data/`.
- Env vars use prefix `DOCUMIND_` (exceptions: none). All config flows through `app.core.config.Settings`; no `os.environ` reads elsewhere.
- Models: generation `qwen2.5:3b`, embeddings `nomic-embed-text` (768-dim), eval judge `qwen2.5:7b`. CPU-only assumptions everywhere.
- No prompts in Python code — all prompt text in `prompts/*.yaml`.
- Corrective loops hard-bounded: max 2 retrieval attempts, max 2 generation attempts total.
- Tests: unit tests run with no services (`uv run pytest -m "not integration"`); integration tests require docker services and run only when `RUN_INTEGRATION=1`.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Library APIs (Phoenix client, CrewAI, LlamaIndex PGVector) move fast — if an installed version's API differs from the code here, adapt at the call site, keep the module's public interface exactly as specified in its task's **Produces** block.

---

### Task 1: Backend scaffold & configuration core

**Files:**
- Create: `backend/pyproject.toml`, `backend/app/__init__.py`, `backend/app/core/__init__.py`, `backend/app/core/config.py`, `backend/app/core/logging.py`, `backend/tests/__init__.py`, `backend/tests/conftest.py`, `backend/tests/unit/test_config.py`, `.env.example`
- Test: `backend/tests/unit/test_config.py`

**Interfaces:**
- Produces: `app.core.config.Settings` (pydantic-settings, env prefix `DOCUMIND_`), `get_settings() -> Settings` (lru_cached), `app.core.logging.setup_logging() -> None`. Every later task imports `get_settings`.

- [ ] **Step 1: Create pyproject and install deps**

`backend/pyproject.toml`:

```toml
[project]
name = "documind-backend"
version = "0.1.0"
description = "DocuMind agentic RAG backend"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
    "pydantic-settings>=2.4",
    "sqlalchemy>=2.0",
    "psycopg[binary]>=3.2",
    "pyyaml>=6.0",
    "httpx>=0.27",
    "python-multipart>=0.0.9",
    "docling>=2.15",
    "llama-index-core>=0.12",
    "llama-index-vector-stores-postgres>=0.4",
    "llama-index-embeddings-ollama>=0.5",
    "llama-index-llms-ollama>=0.5",
    "crewai>=0.95",
    "arize-phoenix-client>=1.0",
    "arize-phoenix-otel>=0.7",
    "openinference-instrumentation-crewai>=0.1",
    "openinference-instrumentation-llama-index>=3.0",
    "openinference-instrumentation-litellm>=0.1",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
eval = [
    "ragas>=0.2.10",
    "langchain-ollama>=0.2",
]

[tool.pytest.ini_options]
markers = ["integration: requires dockerized postgres/ollama/phoenix"]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```

Run: `cd backend && uv sync --all-groups`
Expected: lockfile created, all deps resolve. If a pair conflicts (crewai vs llama-index are the likely pair), relax the floor of the newer one until `uv sync` passes and note the final pins in the commit message.

- [ ] **Step 2: Write the failing config test**

`backend/tests/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

`backend/tests/unit/test_config.py`:

```python
from pathlib import Path


def test_defaults():
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.llm_model == "qwen2.5:3b"
    assert s.embed_model == "nomic-embed-text"
    assert s.embed_dim == 768
    assert s.pipeline_mode == "agentic"
    assert s.max_retrieval_attempts == 2
    assert s.data_dir == Path("data/documents")


def test_env_override(monkeypatch):
    monkeypatch.setenv("DOCUMIND_LLM_MODEL", "llama3.2:3b")
    monkeypatch.setenv("DOCUMIND_PIPELINE_MODE", "simple")
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.llm_model == "llama3.2:3b"
    assert s.pipeline_mode == "simple"


def test_get_settings_cached():
    from app.core.config import get_settings
    assert get_settings() is get_settings()
```

- [ ] **Step 3: Run tests, verify failure**

Run: `uv run pytest tests/unit/test_config.py -v` — Expected: FAIL (ModuleNotFoundError `app.core.config`).

- [ ] **Step 4: Implement config + logging**

`backend/app/core/config.py`:

```python
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_", env_file=".env", extra="ignore"
    )

    # Security
    api_key: str = "documind-dev-key"

    # Services
    database_url: str = "postgresql+psycopg://documind:documind@localhost:5432/documind"
    ollama_base_url: str = "http://localhost:11434"
    phoenix_base_url: str = "http://localhost:6006"

    # Models
    llm_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    judge_model: str = "qwen2.5:7b"
    llm_timeout_seconds: float = 180.0

    # Pipeline
    pipeline_mode: Literal["agentic", "simple"] = "agentic"
    retrieval_top_k: int = 6
    max_retrieval_attempts: int = 2
    max_generation_attempts: int = 2

    # Ingestion
    data_dir: Path = Path("data/documents")
    prompts_dir: Path = Path("prompts")
    chunk_max_tokens: int = 512

    # Vector store
    vector_table_name: str = "rag_chunks"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`backend/app/core/logging.py`:

```python
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:  # idempotent under uvicorn reload
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] [cid=%(correlation_id)s] %(message)s"
    ))
    handler.addFilter(_CorrelationFilter())
    root.addHandler(handler)
    root.setLevel(level)


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            from app.core.correlation import get_correlation_id
            record.correlation_id = get_correlation_id() or "-"
        return True
```

Also create `backend/app/core/correlation.py` now (logging depends on it):

```python
from contextvars import ContextVar

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    _correlation_id.set(value)
```

Create empty `backend/app/__init__.py`, `backend/app/core/__init__.py`, `backend/tests/__init__.py`, `backend/tests/unit/__init__.py`.

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/unit/test_config.py -v` — Expected: 3 PASS.

- [ ] **Step 6: Create `.env.example` at repo root**

```bash
# --- DocuMind backend ---
DOCUMIND_API_KEY=change-me                       # bearer token; OpenWebUI uses it as the OpenAI API key
DOCUMIND_DATABASE_URL=postgresql+psycopg://documind:documind@localhost:5432/documind
DOCUMIND_OLLAMA_BASE_URL=http://localhost:11434
DOCUMIND_PHOENIX_BASE_URL=http://localhost:6006
DOCUMIND_LLM_MODEL=qwen2.5:3b                    # generation model (CPU-friendly)
DOCUMIND_EMBED_MODEL=nomic-embed-text            # 768-dim embeddings
DOCUMIND_JUDGE_MODEL=qwen2.5:7b                  # RAGAs evaluation judge
DOCUMIND_PIPELINE_MODE=agentic                   # agentic | simple (naive RAG bypass)
DOCUMIND_RETRIEVAL_TOP_K=6

# --- docker-compose only ---
POSTGRES_USER=documind
POSTGRES_PASSWORD=documind
POSTGRES_DB=documind
```

- [ ] **Step 7: Commit**

```bash
git add backend .env.example
git commit -m "feat: backend scaffold with typed settings and logging core"
```

---

### Task 2: Docker Compose stack & backend Dockerfile

**Files:**
- Create: `docker-compose.yml`, `backend/Dockerfile`, `backend/.dockerignore`

**Interfaces:**
- Produces: services `postgres:5432`, `ollama:11434`, `phoenix:6006`, `openwebui:3000→8080`, `backend:8000`; named volumes `pgdata`, `ollama_data`, `phoenix_data`, `hf_cache`. Later integration tests assume these host ports.

- [ ] **Step 1: Write compose file**

`docker-compose.yml`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-documind}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-documind}
      POSTGRES_DB: ${POSTGRES_DB:-documind}
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U documind"]
      interval: 5s
      timeout: 3s
      retries: 10

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 5s
      retries: 12

  ollama-init:
    image: ollama/ollama:latest
    depends_on:
      ollama: {condition: service_healthy}
    environment: {OLLAMA_HOST: "http://ollama:11434"}
    entrypoint: >
      sh -c "ollama pull qwen2.5:3b &&
             ollama pull nomic-embed-text &&
             ollama pull qwen2.5:7b"
    restart: "no"

  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["6006:6006", "4317:4317"]
    volumes: [phoenix_data:/mnt/data]
    environment:
      PHOENIX_WORKING_DIR: /mnt/data

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    environment:
      DOCUMIND_DATABASE_URL: postgresql+psycopg://${POSTGRES_USER:-documind}:${POSTGRES_PASSWORD:-documind}@postgres:5432/${POSTGRES_DB:-documind}
      DOCUMIND_OLLAMA_BASE_URL: http://ollama:11434
      DOCUMIND_PHOENIX_BASE_URL: http://phoenix:6006
    volumes:
      - ./data:/app/data
      - ./prompts:/app/prompts
      - hf_cache:/root/.cache
    depends_on:
      postgres: {condition: service_healthy}
      ollama: {condition: service_healthy}

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports: ["3000:8080"]
    environment:
      OPENAI_API_BASE_URL: http://backend:8000/v1
      OPENAI_API_KEY: ${DOCUMIND_API_KEY:-change-me}
      ENABLE_OLLAMA_API: "false"
      WEBUI_AUTH: "false"
    volumes: [openwebui_data:/app/backend/data]
    depends_on: [backend]

volumes:
  pgdata:
  ollama_data:
  phoenix_data:
  hf_cache:
  openwebui_data:
```

- [ ] **Step 2: Write backend Dockerfile**

`backend/Dockerfile`:

```dockerfile
FROM python:3.12-slim

# libgl/libglib needed by docling's layout models (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`backend/.dockerignore`:

```
.venv
tests
__pycache__
*.pyc
```

- [ ] **Step 3: Validate and boot the data-plane services**

Run (repo root): `cp .env.example .env && docker compose config -q && docker compose up -d postgres phoenix ollama ollama-init`
Expected: config valid; `docker compose ps` shows postgres healthy, phoenix running, ollama healthy; `ollama-init` exits 0 after pulling three models (several GB — one-time). Verify: `curl http://localhost:11434/api/tags` lists 3 models; `http://localhost:6006` serves the Phoenix UI.

Note: `backend` will not build/start yet (no `app/main.py` until Task 6) — that is expected; do not start it in this task.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml backend/Dockerfile backend/.dockerignore
git commit -m "feat: docker compose stack (postgres+pgvector, ollama, phoenix, openwebui, backend)"
```

---

### Task 3: Ingestion ledger (DB models + repositories)

**Files:**
- Create: `backend/app/db/__init__.py`, `backend/app/db/models.py`, `backend/app/db/session.py`, `backend/app/db/repository.py`
- Test: `backend/tests/unit/test_repository.py`

**Interfaces:**
- Consumes: `get_settings()` from Task 1.
- Produces:
  - `app.db.models.DocumentRecord` — columns: `id: str` (uuid4 hex, PK), `filename: str`, `sha256: str` (unique), `status: str` (`pending|processing|completed|failed`), `error: str | None`, `page_count: int | None`, `chunk_count: int | None`, `ingested_at: datetime | None`.
  - `app.db.models.IngestJob` — `id: str` (PK), `status: str` (`running|completed|failed`), `total_documents: int`, `completed_documents: int`, `failed_documents: int`, `started_at: datetime`, `finished_at: datetime | None`.
  - `app.db.session.get_engine()`, `get_session()` (context-managed factory), `init_db(engine) -> None` (create_all).
  - `app.db.repository.DocumentRepository(session)` — `get_by_sha(sha)`, `create(filename, sha)`, `mark_processing(id)`, `mark_completed(id, page_count, chunk_count)`, `mark_failed(id, error)`, `list_all()`, `get(id)`, `delete(id)`.
  - `app.db.repository.JobRepository(session)` — `create()`, `get(id)`, `set_total(id, total)`, `update_progress(id, completed, failed)`, `finish(id, status)`.

- [ ] **Step 1: Write failing repository tests** (sqlite in-memory — models must avoid pg-only types)

`backend/tests/unit/test_repository.py`:

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def session():
    from app.db.models import Base
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        yield s


def test_document_lifecycle(session):
    from app.db.repository import DocumentRepository
    repo = DocumentRepository(session)
    doc = repo.create(filename="a.pdf", sha="abc123")
    assert doc.status == "pending"
    assert repo.get_by_sha("abc123").id == doc.id
    repo.mark_processing(doc.id)
    assert repo.get(doc.id).status == "processing"
    repo.mark_completed(doc.id, page_count=3, chunk_count=12)
    got = repo.get(doc.id)
    assert (got.status, got.page_count, got.chunk_count) == ("completed", 3, 12)
    assert got.ingested_at is not None


def test_mark_failed_records_error(session):
    from app.db.repository import DocumentRepository
    repo = DocumentRepository(session)
    doc = repo.create(filename="bad.pdf", sha="ffff")
    repo.mark_failed(doc.id, error="parse error")
    got = repo.get(doc.id)
    assert got.status == "failed" and got.error == "parse error"


def test_job_progress(session):
    from app.db.repository import JobRepository
    repo = JobRepository(session)
    job = repo.create()
    assert job.status == "running"
    repo.update_progress(job.id, completed=2, failed=1)
    repo.finish(job.id, status="completed")
    got = repo.get(job.id)
    assert (got.completed_documents, got.failed_documents, got.status) == (2, 1, "completed")
    assert got.finished_at is not None
```

- [ ] **Step 2: Run — verify FAIL** (`uv run pytest tests/unit/test_repository.py -v` → ModuleNotFoundError)

- [ ] **Step 3: Implement models, session, repositories**

`backend/app/db/models.py`:

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IngestJob(Base):
    __tablename__ = "ingest_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(16), default="running")
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    completed_documents: Mapped[int] = mapped_column(Integer, default=0)
    failed_documents: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

`backend/app/db/session.py`:

```python
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Base


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def init_db(engine: Engine | None = None) -> None:
    Base.metadata.create_all(engine or get_engine())


@contextmanager
def get_session() -> Iterator[Session]:
    factory = sessionmaker(bind=get_engine())
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

`backend/app/db/repository.py`:

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentRecord, IngestJob, utcnow


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, filename: str, sha: str) -> DocumentRecord:
        doc = DocumentRecord(filename=filename, sha256=sha)
        self.session.add(doc)
        self.session.flush()
        return doc

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self.session.get(DocumentRecord, doc_id)

    def get_by_sha(self, sha: str) -> DocumentRecord | None:
        return self.session.scalar(select(DocumentRecord).where(DocumentRecord.sha256 == sha))

    def list_all(self) -> list[DocumentRecord]:
        return list(self.session.scalars(select(DocumentRecord).order_by(DocumentRecord.filename)))

    def mark_processing(self, doc_id: str) -> None:
        self.get(doc_id).status = "processing"
        self.session.flush()

    def mark_completed(self, doc_id: str, page_count: int, chunk_count: int) -> None:
        doc = self.get(doc_id)
        doc.status, doc.page_count, doc.chunk_count = "completed", page_count, chunk_count
        doc.error, doc.ingested_at = None, utcnow()
        self.session.flush()

    def mark_failed(self, doc_id: str, error: str) -> None:
        doc = self.get(doc_id)
        doc.status, doc.error = "failed", error[:2000]
        self.session.flush()

    def delete(self, doc_id: str) -> None:
        doc = self.get(doc_id)
        if doc is not None:
            self.session.delete(doc)
            self.session.flush()


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self) -> IngestJob:
        job = IngestJob()
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: str) -> IngestJob | None:
        return self.session.get(IngestJob, job_id)

    def set_total(self, job_id: str, total: int) -> None:
        self.get(job_id).total_documents = total
        self.session.flush()

    def update_progress(self, job_id: str, completed: int, failed: int) -> None:
        job = self.get(job_id)
        job.completed_documents, job.failed_documents = completed, failed
        self.session.flush()

    def finish(self, job_id: str, status: str) -> None:
        job = self.get(job_id)
        job.status, job.finished_at = status, utcnow()
        self.session.flush()
```

Create empty `backend/app/db/__init__.py`.

- [ ] **Step 4: Run — verify PASS** (`uv run pytest tests/unit/test_repository.py -v` → 3 PASS)

- [ ] **Step 5: Commit**

```bash
git add backend/app/db backend/tests/unit/test_repository.py
git commit -m "feat: ingestion ledger models and repositories"
```

---

### Task 4: Docling preprocessing & chunk contextualization

**Files:**
- Create: `backend/app/ingestion/__init__.py`, `backend/app/ingestion/types.py`, `backend/app/ingestion/preprocessor.py`, `backend/app/ingestion/contextualizer.py`
- Test: `backend/tests/unit/test_contextualizer.py`, `backend/tests/integration/test_preprocessor.py`

**Interfaces:**
- Produces:
  - `app.ingestion.types.RawChunk` — pydantic model: `text: str`, `section_path: list[str]`, `pages: list[int]`, `is_table: bool`.
  - `app.ingestion.types.ParsedDocument` — `title: str`, `page_count: int`, `chunks: list[RawChunk]`.
  - `app.ingestion.preprocessor.parse_pdf(path: Path, max_tokens: int = 512) -> ParsedDocument` (Docling convert + HybridChunker).
  - `app.ingestion.contextualizer.contextualize(chunk: RawChunk, doc_title: str) -> str` — returns `[{title} > {sections}]\n\n{text}` (`| table` marker for table chunks).

- [ ] **Step 1: Write failing contextualizer tests**

`backend/tests/unit/test_contextualizer.py`:

```python
from app.ingestion.types import RawChunk


def _chunk(**kw):
    base = dict(text="Total amount due: EUR 1,200", section_path=[], pages=[1], is_table=False)
    base.update(kw)
    return RawChunk(**base)


def test_header_with_sections():
    from app.ingestion.contextualizer import contextualize
    out = contextualize(_chunk(section_path=["Invoice Details", "Line Items"]), "Invoice June 2026")
    assert out.startswith("[Invoice June 2026 > Invoice Details > Line Items]\n\n")
    assert out.endswith("Total amount due: EUR 1,200")


def test_header_without_sections_uses_title_only():
    from app.ingestion.contextualizer import contextualize
    out = contextualize(_chunk(), "Payslip")
    assert out.startswith("[Payslip]\n\n")


def test_table_chunk_flagged():
    from app.ingestion.contextualizer import contextualize
    out = contextualize(_chunk(is_table=True, section_path=["Earnings"]), "Payslip")
    assert out.startswith("[Payslip > Earnings | table]\n\n")
```

- [ ] **Step 2: Run — verify FAIL** (`uv run pytest tests/unit/test_contextualizer.py -v`)

- [ ] **Step 3: Implement types + contextualizer**

`backend/app/ingestion/types.py`:

```python
from pydantic import BaseModel, Field


class RawChunk(BaseModel):
    text: str
    section_path: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    is_table: bool = False


class ParsedDocument(BaseModel):
    title: str
    page_count: int
    chunks: list[RawChunk]
```

`backend/app/ingestion/contextualizer.py`:

```python
from app.ingestion.types import RawChunk


def contextualize(chunk: RawChunk, doc_title: str) -> str:
    parts = [doc_title, *chunk.section_path]
    header = " > ".join(p.strip() for p in parts if p and p.strip())
    if chunk.is_table:
        header += " | table"
    return f"[{header}]\n\n{chunk.text}"
```

- [ ] **Step 4: Run — verify PASS** (3 PASS)

- [ ] **Step 5: Implement the Docling preprocessor**

`backend/app/ingestion/preprocessor.py`:

```python
import logging
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from app.ingestion.types import ParsedDocument, RawChunk

logger = logging.getLogger(__name__)

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:  # heavyweight: loads layout models on first use
        _converter = DocumentConverter()
    return _converter


def parse_pdf(path: Path, max_tokens: int = 512) -> ParsedDocument:
    result = _get_converter().convert(str(path))
    doc = result.document
    title = (doc.name or path.stem).strip() or path.stem
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)

    chunks: list[RawChunk] = []
    for chunk in chunker.chunk(doc):
        meta = chunk.meta
        headings = list(getattr(meta, "headings", None) or [])
        pages: set[int] = set()
        is_table = False
        for item in getattr(meta, "doc_items", None) or []:
            if type(item).__name__ == "TableItem":
                is_table = True
            for prov in getattr(item, "prov", None) or []:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    pages.add(int(page_no))
        text = chunk.text.strip()
        if not text:
            continue
        chunks.append(RawChunk(
            text=text, section_path=headings, pages=sorted(pages), is_table=is_table
        ))

    page_count = len(getattr(doc, "pages", {}) or {})
    logger.info("parsed %s: %d pages, %d chunks", path.name, page_count, len(chunks))
    return ParsedDocument(title=title, page_count=page_count, chunks=chunks)
```

- [ ] **Step 6: Write the marked integration test**

`backend/tests/integration/__init__.py` (empty) and `backend/tests/integration/test_preprocessor.py`:

```python
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DOCS = Path(__file__).resolve().parents[3] / "data" / "documents"


@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="integration disabled")
def test_parse_real_pdf():
    pdfs = sorted(DOCS.glob("*.pdf"))
    if not pdfs:
        pytest.skip("no local PDFs")
    from app.ingestion.preprocessor import parse_pdf
    parsed = parse_pdf(pdfs[0])
    assert parsed.page_count >= 1
    assert len(parsed.chunks) >= 1
    assert all(c.text.strip() for c in parsed.chunks)
```

Note: `parents[3]` from `backend/tests/integration/test_preprocessor.py` resolves to the repo root `documind/`.

Run: `uv run pytest tests -m "not integration" -v` — Expected: all unit tests still PASS (integration auto-skipped). Optionally `RUN_INTEGRATION=1 uv run pytest tests/integration/test_preprocessor.py -v` — first run downloads Docling layout models (takes minutes).

- [ ] **Step 7: Commit**

```bash
git add backend/app/ingestion backend/tests
git commit -m "feat: docling preprocessing with structure-aware chunking and contextualization"
```

---

### Task 5: Vector store, embedding & ingestion pipeline

**Files:**
- Create: `backend/app/retrieval/__init__.py`, `backend/app/retrieval/vector_store.py`, `backend/app/ingestion/pipeline.py`, `scripts/ingest.py`
- Test: `backend/tests/unit/test_ingestion_pipeline.py`, `backend/tests/integration/test_ingest_e2e.py`

**Interfaces:**
- Consumes: Task 3 repositories, Task 4 `parse_pdf`/`contextualize`.
- Produces:
  - `app.retrieval.vector_store.get_vector_store() -> PGVectorStore` (hybrid_search=True, table `rag_chunks`, embed_dim from settings; lru_cached).
  - `app.retrieval.vector_store.get_embed_model() -> OllamaEmbedding` (lru_cached).
  - `app.ingestion.pipeline.IngestionPipeline(parse=parse_pdf, store=None, embed=None)` — injectable for tests; method `run(job_id: str) -> None` scans `settings.data_dir`, and per file: sha-check → ledger → parse → contextualize → embed → index; method `ingest_file(path: Path) -> str` (returns document id, raises on failure); `delete_document(doc_id: str) -> None` removes chunks (by `ref_doc_id`) + ledger row.
  - Chunk node metadata keys (used by retrieval + citations): `doc_id`, `title`, `section_path` (" > " joined), `pages` (list[int]), `is_table` (bool), `filename`.

- [ ] **Step 1: Write failing pipeline unit tests** (mock parse + store + embeddings; sqlite ledger)

`backend/tests/unit/test_ingestion_pipeline.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.types import ParsedDocument, RawChunk


@pytest.fixture
def session_factory():
    from app.db.models import Base
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _parsed():
    return ParsedDocument(title="Doc", page_count=2, chunks=[
        RawChunk(text="alpha", section_path=["S1"], pages=[1]),
        RawChunk(text="beta", section_path=["S2"], pages=[2], is_table=True),
    ])


def _pipeline(session_factory, tmp_path, parse=None):
    from app.ingestion.pipeline import IngestionPipeline
    store, embed = MagicMock(), MagicMock()
    embed.get_text_embedding_batch.return_value = [[0.1] * 768, [0.2] * 768]
    p = IngestionPipeline(
        parse=parse or MagicMock(return_value=_parsed()),
        store=store, embed=embed,
        session_factory=session_factory, data_dir=tmp_path,
    )
    return p, store


def test_ingest_file_indexes_contextualized_chunks(session_factory, tmp_path):
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-fake")
    pipeline, store = _pipeline(session_factory, tmp_path)
    doc_id = pipeline.ingest_file(pdf)

    nodes = store.add.call_args.args[0]
    assert len(nodes) == 2
    assert nodes[0].text.startswith("[Doc > S1]\n\n")
    assert nodes[0].embedding == [0.1] * 768
    assert nodes[0].metadata["doc_id"] == doc_id
    assert nodes[1].metadata["is_table"] is True

    from app.db.repository import DocumentRepository
    with session_factory() as s:
        rec = DocumentRepository(s).get(doc_id)
        assert (rec.status, rec.chunk_count, rec.page_count) == ("completed", 2, 2)


def test_run_skips_unchanged_and_isolates_failures(session_factory, tmp_path):
    good, bad = tmp_path / "good.pdf", tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF-1"); bad.write_bytes(b"%PDF-2")

    def parse(path, max_tokens=512):
        if "bad" in str(path):
            raise ValueError("corrupt pdf")
        return _parsed()

    pipeline, _ = _pipeline(session_factory, tmp_path, parse=MagicMock(side_effect=parse))
    from app.db.repository import JobRepository
    with session_factory() as s:
        job_id = JobRepository(s).create().id; s.commit()

    pipeline.run(job_id)

    from app.db.repository import DocumentRepository
    with session_factory() as s:
        docs = {d.filename: d for d in DocumentRepository(s).list_all()}
        job = JobRepository(s).get(job_id)
    assert docs["good.pdf"].status == "completed"
    assert docs["bad.pdf"].status == "failed" and "corrupt" in docs["bad.pdf"].error
    assert (job.completed_documents, job.failed_documents, job.status) == (1, 1, "completed")

    # second run: same hashes -> both skipped, no new parse calls for good.pdf
    calls_before = pipeline.parse.call_count
    with session_factory() as s:
        job2 = JobRepository(s).create().id; s.commit()
    pipeline.run(job2)
    assert pipeline.parse.call_count == calls_before + 1  # only failed doc retried
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implement vector store factory**

`backend/app/retrieval/vector_store.py`:

```python
from functools import lru_cache
from urllib.parse import urlparse

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import get_settings


@lru_cache
def get_vector_store() -> PGVectorStore:
    s = get_settings()
    url = urlparse(s.database_url.replace("postgresql+psycopg", "postgresql"))
    return PGVectorStore.from_params(
        database=url.path.lstrip("/"),
        host=url.hostname,
        port=str(url.port or 5432),
        user=url.username,
        password=url.password,
        table_name=s.vector_table_name,   # llama-index stores it as data_rag_chunks
        embed_dim=s.embed_dim,
        hybrid_search=True,
        text_search_config="english",
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64,
                     "hnsw_ef_search": 40, "hnsw_dist_method": "vector_cosine_ops"},
    )


@lru_cache
def get_embed_model() -> OllamaEmbedding:
    s = get_settings()
    return OllamaEmbedding(model_name=s.embed_model, base_url=s.ollama_base_url)
```

- [ ] **Step 4: Implement the ingestion pipeline**

`backend/app/ingestion/pipeline.py`:

```python
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.repository import DocumentRepository, JobRepository
from app.ingestion.contextualizer import contextualize
from app.ingestion.types import ParsedDocument

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        parse: Callable[..., ParsedDocument] | None = None,
        store=None,
        embed=None,
        session_factory: sessionmaker | None = None,
        data_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        if parse is None:
            from app.ingestion.preprocessor import parse_pdf
            parse = parse_pdf
        if store is None:
            from app.retrieval.vector_store import get_vector_store
            store = get_vector_store()
        if embed is None:
            from app.retrieval.vector_store import get_embed_model
            embed = get_embed_model()
        if session_factory is None:
            from app.db.session import get_engine
            session_factory = sessionmaker(bind=get_engine())
        self.parse, self.store, self.embed = parse, store, embed
        self.session_factory = session_factory
        self.data_dir = data_dir or settings.data_dir
        self.chunk_max_tokens = settings.chunk_max_tokens

    def run(self, job_id: str) -> None:
        pdfs = sorted(self.data_dir.glob("*.pdf"))
        completed = failed = 0
        with self.session_factory() as s:
            JobRepository(s).set_total(job_id, len(pdfs))
            s.commit()
        for path in pdfs:
            try:
                if self._already_ingested(path):
                    logger.info("skipping unchanged %s", path.name)
                    continue
                self.ingest_file(path)
                completed += 1
            except Exception as exc:  # per-document isolation
                logger.exception("ingestion failed for %s", path.name)
                failed += 1
                self._mark_failed(path, str(exc))
            with self.session_factory() as s:
                JobRepository(s).update_progress(job_id, completed, failed)
                s.commit()
        with self.session_factory() as s:
            JobRepository(s).finish(job_id, "completed")
            s.commit()

    def _already_ingested(self, path: Path) -> bool:
        sha = _sha256(path)
        with self.session_factory() as s:
            existing = DocumentRepository(s).get_by_sha(sha)
            return existing is not None and existing.status == "completed"

    def _mark_failed(self, path: Path, error: str) -> None:
        sha = _sha256(path)
        with self.session_factory() as s:
            repo = DocumentRepository(s)
            doc = repo.get_by_sha(sha) or repo.create(filename=path.name, sha=sha)
            repo.mark_failed(doc.id, error)
            s.commit()

    def ingest_file(self, path: Path) -> str:
        sha = _sha256(path)
        with self.session_factory() as s:
            repo = DocumentRepository(s)
            doc = repo.get_by_sha(sha) or repo.create(filename=path.name, sha=sha)
            doc_id = doc.id
            repo.mark_processing(doc_id)
            s.commit()

        parsed = self.parse(path, max_tokens=self.chunk_max_tokens)
        texts = [contextualize(c, parsed.title) for c in parsed.chunks]
        embeddings = self.embed.get_text_embedding_batch(texts, show_progress=False)

        nodes = []
        for chunk, text, emb in zip(parsed.chunks, texts, embeddings):
            nodes.append(TextNode(
                text=text,
                embedding=emb,
                metadata={
                    "doc_id": doc_id,
                    "title": parsed.title,
                    "filename": path.name,
                    "section_path": " > ".join(chunk.section_path),
                    "pages": chunk.pages,
                    "is_table": chunk.is_table,
                },
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)},
            ))
        self.store.delete(ref_doc_id=doc_id)  # re-ingest safety: drop stale chunks
        self.store.add(nodes)

        with self.session_factory() as s:
            DocumentRepository(s).mark_completed(doc_id, parsed.page_count, len(nodes))
            s.commit()
        logger.info("ingested %s: %d chunks", path.name, len(nodes))
        return doc_id

    def delete_document(self, doc_id: str) -> None:
        self.store.delete(ref_doc_id=doc_id)
        with self.session_factory() as s:
            DocumentRepository(s).delete(doc_id)
            s.commit()
```

- [ ] **Step 5: Run — verify PASS** (`uv run pytest tests/unit/test_ingestion_pipeline.py -v` → 2 PASS)

- [ ] **Step 6: CLI script + integration test**

`scripts/ingest.py` (repo root; run as `uv run --project backend python scripts/ingest.py`):

```python
"""Ingest all PDFs from data/documents into the vector store."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.logging import setup_logging          # noqa: E402
from app.db.repository import JobRepository         # noqa: E402
from app.db.session import get_session, init_db     # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402


def main() -> int:
    setup_logging()
    init_db()
    with get_session() as s:
        job_id = JobRepository(s).create().id
    IngestionPipeline().run(job_id)
    with get_session() as s:
        job = JobRepository(s).get(job_id)
        print(f"job {job_id}: {job.completed_documents} ok, {job.failed_documents} failed")
        return 1 if job.failed_documents else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`backend/tests/integration/test_ingest_e2e.py`:

```python
import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="integration disabled"),
]


def test_full_ingest_and_ledger():
    from app.db.repository import JobRepository
    from app.db.session import get_session, init_db
    from app.ingestion.pipeline import IngestionPipeline

    init_db()
    with get_session() as s:
        job_id = JobRepository(s).create().id
    pipeline = IngestionPipeline()
    if not sorted(pipeline.data_dir.glob("*.pdf")):
        pytest.skip("no local PDFs")
    pipeline.run(job_id)
    with get_session() as s:
        job = JobRepository(s).get(job_id)
        assert job.status == "completed"
        assert job.completed_documents >= 1
```

Run against the live stack (services from Task 2 up): `RUN_INTEGRATION=1 uv run pytest tests/integration/test_ingest_e2e.py -v` — Expected: PASS; then `docker compose exec postgres psql -U documind -c "select count(*) from data_rag_chunks;"` shows > 0 rows. (First real run is slow: docling model download + CPU embedding.)

- [ ] **Step 7: Commit**

```bash
git add backend/app/retrieval backend/app/ingestion/pipeline.py scripts/ingest.py backend/tests
git commit -m "feat: pgvector hybrid store and idempotent ingestion pipeline with CLI"
```

---

### Task 6: FastAPI app, auth & ingestion/documents/health APIs

**Files:**
- Create: `backend/app/core/auth.py`, `backend/app/main.py`, `backend/app/api/__init__.py`, `backend/app/api/schemas.py`, `backend/app/api/documents.py`, `backend/app/api/ingest.py`, `backend/app/api/health.py`
- Test: `backend/tests/unit/test_api_documents.py`

**Interfaces:**
- Consumes: Tasks 3 & 5.
- Produces:
  - `app.main.create_app() -> FastAPI` and module-level `app` (uvicorn target `app.main:app`). Startup: `setup_logging()`, `init_db()`, `setup_tracing()` (no-op stub until Task 9), prompt sync (no-op until Task 8).
  - `app.core.auth.require_api_key` — FastAPI dependency; 401 unless header `Authorization: Bearer <DOCUMIND_API_KEY>`. `GET /health` is the only unauthenticated route.
  - Routes per spec §4.2: `POST /api/v1/documents` (multipart upload → saves into `data_dir`, ingests synchronously in a worker thread job), `GET /api/v1/documents`, `GET /api/v1/documents/{id}`, `DELETE /api/v1/documents/{id}`, `POST /api/v1/ingest` → `{"job_id": ...}`, `GET /api/v1/ingest/{job_id}`, `GET /health`.
  - `app.api.schemas` — pydantic response models: `DocumentOut(id, filename, status, error, page_count, chunk_count, ingested_at)`, `JobOut(id, status, total_documents, completed_documents, failed_documents, started_at, finished_at)`, `HealthOut(status, postgres, ollama, phoenix)`.

- [ ] **Step 1: Write failing API tests** (TestClient; pipeline mocked; sqlite ledger via monkeypatched engine)

`backend/tests/unit/test_api_documents.py`:

```python
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

AUTH = {"Authorization": "Bearer documind-dev-key"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.db.session as db_session
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    db_session.get_engine.cache_clear()
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(tmp_path))
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


def test_health_is_public(client):
    with patch("app.api.health._check_url", return_value=True):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_requires_auth(client):
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_ingest_creates_job_and_reports_status(client):
    with patch("app.api.ingest.IngestionPipeline") as pipe_cls:
        pipe_cls.return_value.run = MagicMock()
        r = client.post("/api/v1/ingest", headers=AUTH)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    r2 = client.get(f"/api/v1/ingest/{job_id}", headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["id"] == job_id


def test_upload_saves_pdf_and_lists_document(client, tmp_path):
    with patch("app.api.documents.IngestionPipeline") as pipe_cls:
        pipe_cls.return_value.ingest_file = MagicMock(return_value="doc123")
        r = client.post(
            "/api/v1/documents", headers=AUTH,
            files={"file": ("new.pdf", b"%PDF-1.7 fake", "application/pdf")},
        )
    assert r.status_code == 201
    assert (tmp_path / "new.pdf").exists()


def test_upload_rejects_non_pdf(client):
    r = client.post(
        "/api/v1/documents", headers=AUTH,
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implement auth, schemas, routers, app factory**

`backend/app/core/auth.py`:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if credentials is None or credentials.credentials != get_settings().api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
```

`backend/app/api/schemas.py`:

```python
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    error: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    ingested_at: datetime | None = None
    model_config = {"from_attributes": True}


class JobOut(BaseModel):
    id: str
    status: str
    total_documents: int
    completed_documents: int
    failed_documents: int
    started_at: datetime
    finished_at: datetime | None = None
    model_config = {"from_attributes": True}


class HealthOut(BaseModel):
    status: str
    postgres: bool
    ollama: bool
    phoenix: bool
```

`backend/app/api/documents.py`:

```python
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import DocumentOut
from app.core.config import get_settings
from app.db.repository import DocumentRepository
from app.db.session import get_session
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    with get_session() as s:
        return [DocumentOut.model_validate(d) for d in DocumentRepository(s).list_all()]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str) -> DocumentOut:
    with get_session() as s:
        doc = DocumentRepository(s).get(doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        return DocumentOut.model_validate(doc)


@router.post("", response_model=DocumentOut, status_code=201)
def upload_document(file: UploadFile) -> DocumentOut:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.data_dir / Path(file.filename).name
    target.write_bytes(file.file.read())
    logger.info("uploaded %s", target.name)
    doc_id = IngestionPipeline().ingest_file(target)
    with get_session() as s:
        return DocumentOut.model_validate(DocumentRepository(s).get(doc_id))


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    with get_session() as s:
        if DocumentRepository(s).get(doc_id) is None:
            raise HTTPException(404, "document not found")
    IngestionPipeline().delete_document(doc_id)
```

`backend/app/api/ingest.py`:

```python
import logging
import threading

from fastapi import APIRouter, HTTPException

from app.api.schemas import JobOut
from app.db.repository import JobRepository
from app.db.session import get_session
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.post("", status_code=202)
def start_ingest() -> dict:
    with get_session() as s:
        job_id = JobRepository(s).create().id

    def _run() -> None:
        try:
            IngestionPipeline().run(job_id)
        except Exception:
            logger.exception("ingest job %s crashed", job_id)
            with get_session() as s:
                JobRepository(s).finish(job_id, "failed")

    threading.Thread(target=_run, name=f"ingest-{job_id}", daemon=True).start()
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=JobOut)
def job_status(job_id: str) -> JobOut:
    with get_session() as s:
        job = JobRepository(s).get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return JobOut.model_validate(job)
```

`backend/app/api/health.py`:

```python
import httpx
from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import HealthOut
from app.core.config import get_settings

router = APIRouter(tags=["health"])


def _check_url(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3.0).status_code < 500
    except httpx.HTTPError:
        return False


def _check_db() -> bool:
    try:
        from app.db.session import get_engine
        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    s = get_settings()
    pg, ol, ph = _check_db(), _check_url(f"{s.ollama_base_url}/api/tags"), _check_url(s.phoenix_base_url)
    return HealthOut(status="ok" if (pg and ol) else "degraded", postgres=pg, ollama=ol, phoenix=ph)
```

`backend/app/main.py`:

```python
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.core.auth import require_api_key
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    from app.db.session import init_db
    init_db()
    try:  # tracing + prompt sync are best-effort (real impls in Tasks 8-9)
        from app.observability.tracing import setup_tracing
        setup_tracing()
    except ImportError:
        logger.info("tracing module not present yet")
    try:
        from app.observability.prompts import get_prompt_manager
        get_prompt_manager().sync_to_phoenix()
    except ImportError:
        logger.info("prompt manager not present yet")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuMind API",
        version="1.0.0",
        description="Agentic RAG document search platform",
        lifespan=lifespan,
    )
    from app.api import documents, health, ingest
    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    app.include_router(documents.router, dependencies=protected)
    app.include_router(ingest.router, dependencies=protected)
    return app


app = create_app()
```

Create empty `backend/app/api/__init__.py`.

- [ ] **Step 4: Run — verify PASS** (`uv run pytest tests/unit/test_api_documents.py -v` → 5 PASS; also full suite `uv run pytest -m "not integration"`)

- [ ] **Step 5: Commit**

```bash
git add backend/app backend/tests
git commit -m "feat: FastAPI app with auth, document/ingest/health APIs"
```

---

### Task 7: Prompt externalization (YAML + Phoenix PromptOps)

**Files:**
- Create: `prompts/router.yaml`, `prompts/rewriter.yaml`, `prompts/grader.yaml`, `prompts/synthesizer.yaml`, `prompts/hallucination_checker.yaml`, `backend/app/observability/__init__.py`, `backend/app/observability/prompts.py`
- Test: `backend/tests/unit/test_prompts.py`

**Interfaces:**
- Consumes: `get_settings()`.
- Produces: `app.observability.prompts.PromptManager` with `get(name: str, **variables) -> str` (formats `{var}` placeholders), `sync_to_phoenix() -> None` (best-effort, logs on failure), `get_prompt_manager() -> PromptManager` (lru_cached). Phoenix-pulled template wins over YAML when available; YAML is the fallback. Prompt names used by later tasks: `router`, `rewriter`, `grader`, `synthesizer`, `hallucination_checker`.

- [ ] **Step 1: Write the five prompt YAML files** (repo-root `prompts/`; format: `name`, `version`, `description`, `template`)

`prompts/router.yaml`:

```yaml
name: router
version: 1
description: Classify whether a user message needs document retrieval.
template: |
  You are a query router for a document search assistant. The knowledge base
  contains business documents (invoices, payslips, filed forms, timesheets).

  Classify the user message:
  - "rag"    -> the answer may come from the documents
  - "direct" -> greeting, small talk, or a question about you/the assistant

  User message: {question}

  Reply with exactly one word: rag or direct. No other text.
```

`prompts/rewriter.yaml`:

```yaml
name: rewriter
version: 1
description: Rewrite a chat question into standalone search queries.
template: |
  You reformulate user questions into standalone search queries for a
  document knowledge base (invoices, payslips, forms, timesheets).

  Conversation so far (may be empty):
  {history}

  User question: {question}
  {feedback}

  Write 1 to 3 short, standalone search queries that together cover the
  question. Resolve pronouns using the conversation. Reply ONLY with a JSON
  array of strings, e.g. ["query one", "query two"].
```

`prompts/grader.yaml`:

```yaml
name: grader
version: 1
description: Grade retrieved chunks for relevance to the question.
template: |
  You are a strict relevance grader. For each numbered context chunk below,
  decide if it helps answer the question.

  Question: {question}

  Chunks:
  {chunks}

  Reply ONLY with a JSON array of the numbers of relevant chunks, e.g. [0, 2].
  Reply [] if none are relevant.
```

`prompts/synthesizer.yaml`:

```yaml
name: synthesizer
version: 1
description: Write a grounded, cited answer from context.
template: |
  You are DocuMind, a document search assistant. Answer the question using
  ONLY the context below. Rules:
  - Cite sources inline as [title, section] after each claim.
  - Quote numbers, dates, and amounts exactly as written in the context.
  - If the context does not contain the answer, say so plainly and do not guess.
  {feedback}

  Context:
  {context}

  Question: {question}

  Answer:
```

`prompts/hallucination_checker.yaml`:

```yaml
name: hallucination_checker
version: 1
description: Verify an answer is grounded in the retrieved context.
template: |
  You are a groundedness checker. Verify every factual claim in the answer
  is supported by the context. An answer that says the information is not
  available is grounded.

  Context:
  {context}

  Answer to verify:
  {answer}

  Reply with exactly one word: yes if fully grounded, no otherwise.
```

- [ ] **Step 2: Write failing prompt manager tests**

`backend/tests/unit/test_prompts.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def _manager(**kw):
    from app.observability.prompts import PromptManager
    return PromptManager(prompts_dir=PROMPTS_DIR, **kw)


def test_loads_and_formats_yaml():
    m = _manager()
    out = m.get("router", question="hello there")
    assert "hello there" in out and "{question}" not in out


def test_all_five_prompts_load():
    m = _manager()
    for name in ["router", "rewriter", "grader", "synthesizer", "hallucination_checker"]:
        assert m.template(name)


def test_phoenix_override_wins():
    m = _manager()
    m._phoenix_templates["router"] = "OVERRIDE {question}"
    assert m.get("router", question="x") == "OVERRIDE x"


def test_phoenix_failure_falls_back_to_yaml():
    m = _manager()
    m._client = MagicMock()
    m._client.prompts.get.side_effect = RuntimeError("phoenix down")
    m.refresh_from_phoenix()          # must not raise
    assert "rag" in m.get("router", question="q")
```

- [ ] **Step 3: Run — verify FAIL**

- [ ] **Step 4: Implement PromptManager**

`backend/app/observability/prompts.py`:

```python
import logging
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.config import get_settings

logger = logging.getLogger(__name__)

PROMPT_NAMES = ["router", "rewriter", "grader", "synthesizer", "hallucination_checker"]


class PromptManager:
    """YAML-backed prompts, optionally overridden by Phoenix's prompt hub.

    Phoenix is best-effort: sync/refresh failures are logged, never raised,
    and the git-versioned YAML files remain the source of truth fallback.
    """

    def __init__(self, prompts_dir: Path | None = None, client=None) -> None:
        self.prompts_dir = prompts_dir or get_settings().prompts_dir
        self._yaml: dict[str, dict] = {}
        self._phoenix_templates: dict[str, str] = {}
        self._client = client
        self._load_yaml()

    def _load_yaml(self) -> None:
        for name in PROMPT_NAMES:
            path = self.prompts_dir / f"{name}.yaml"
            with path.open(encoding="utf-8") as f:
                self._yaml[name] = yaml.safe_load(f)

    def template(self, name: str) -> str:
        if name in self._phoenix_templates:
            return self._phoenix_templates[name]
        return self._yaml[name]["template"]

    def version(self, name: str) -> str:
        source = "phoenix" if name in self._phoenix_templates else "yaml"
        return f"{source}:v{self._yaml[name].get('version', 1)}"

    def get(self, name: str, **variables: str) -> str:
        text = self.template(name)
        for key, value in variables.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    # --- Phoenix integration (best-effort) ---

    def _phoenix_client(self):
        if self._client is None:
            from phoenix.client import Client
            self._client = Client(base_url=get_settings().phoenix_base_url)
        return self._client

    def sync_to_phoenix(self) -> None:
        """Register YAML prompts in Phoenix so they can be edited in its UI."""
        try:
            from phoenix.client.types import PromptVersion
            client = self._phoenix_client()
            for name, data in self._yaml.items():
                client.prompts.create(
                    name=name,
                    prompt_description=data.get("description", ""),
                    version=PromptVersion(
                        [{"role": "system", "content": data["template"]}],
                        model_name=get_settings().llm_model,
                    ),
                )
            logger.info("synced %d prompts to phoenix", len(self._yaml))
        except Exception as exc:
            logger.warning("phoenix prompt sync skipped: %s", exc)

    def refresh_from_phoenix(self) -> None:
        """Pull latest prompt versions from Phoenix (UI edits win over YAML)."""
        try:
            client = self._phoenix_client()
            for name in PROMPT_NAMES:
                prompt = client.prompts.get(prompt_identifier=name)
                messages = prompt.format().get("messages", [])
                if messages:
                    self._phoenix_templates[name] = messages[0]["content"]
        except Exception as exc:
            logger.warning("phoenix prompt refresh skipped: %s", exc)


@lru_cache
def get_prompt_manager() -> PromptManager:
    return PromptManager()
```

Note: the Phoenix client API shape (`prompts.create` / `PromptVersion` / `prompt.format()`) is version-sensitive — adapt call sites to the installed `arize-phoenix-client`, keep `get`/`template`/`sync_to_phoenix`/`refresh_from_phoenix` signatures stable.

- [ ] **Step 5: Run — verify PASS** (`uv run pytest tests/unit/test_prompts.py -v` → 4 PASS)

- [ ] **Step 6: Commit**

```bash
git add prompts backend/app/observability backend/tests/unit/test_prompts.py
git commit -m "feat: externalized YAML prompts with Phoenix PromptOps sync"
```

---

### Task 8: Observability — Phoenix tracing & correlation IDs

**Files:**
- Create: `backend/app/observability/tracing.py`, `backend/app/core/middleware.py`
- Modify: `backend/app/main.py` (add middleware registration in `create_app`)
- Test: `backend/tests/unit/test_middleware.py`

**Interfaces:**
- Consumes: `app.core.correlation` (Task 1), `get_settings()`.
- Produces:
  - `app.observability.tracing.setup_tracing() -> None` — registers a TracerProvider exporting OTLP-HTTP to `{phoenix_base_url}/v1/traces` (project `documind`), then instruments CrewAI, LlamaIndex and LiteLLM via OpenInference. Idempotent; failures logged, never raised.
  - `app.core.middleware.CorrelationIdMiddleware` — reads `X-Correlation-ID` or generates uuid4 hex; sets contextvar; echoes header on response; stamps it as `correlation.id` attribute on the current OTel span.

- [ ] **Step 1: Write failing middleware test**

`backend/tests/unit/test_middleware.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app():
    from app.core.middleware import CorrelationIdMiddleware
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping():
        from app.core.correlation import get_correlation_id
        return {"cid": get_correlation_id()}

    return app


def test_generates_correlation_id():
    r = TestClient(_app()).get("/ping")
    assert r.headers["x-correlation-id"]
    assert r.json()["cid"] == r.headers["x-correlation-id"]


def test_respects_incoming_correlation_id():
    r = TestClient(_app()).get("/ping", headers={"X-Correlation-ID": "abc-123"})
    assert r.headers["x-correlation-id"] == "abc-123"
    assert r.json()["cid"] == "abc-123"
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implement middleware + tracing setup**

`backend/app/core/middleware.py`:

```python
import uuid

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.correlation import set_correlation_id


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        set_correlation_id(cid)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("correlation.id", cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response
```

`backend/app/observability/tracing.py`:

```python
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_initialized = False


def setup_tracing() -> None:
    global _initialized
    if _initialized:
        return
    try:
        from phoenix.otel import register
        tracer_provider = register(
            project_name="documind",
            endpoint=f"{get_settings().phoenix_base_url}/v1/traces",
            batch=True,
            set_global_tracer_provider=True,
        )
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        _initialized = True
        logger.info("phoenix tracing initialized")
    except Exception as exc:
        logger.warning("tracing setup skipped: %s", exc)
```

In `create_app()` (Task 6 file), after building `app` add:

```python
    from app.core.middleware import CorrelationIdMiddleware
    app.add_middleware(CorrelationIdMiddleware)
```

- [ ] **Step 4: Run — verify PASS** (2 PASS; full unit suite still green)

- [ ] **Step 5: Live check** — with the stack up, `docker compose up -d --build backend`, hit `/health`, open Phoenix at `http://localhost:6006` → project `documind` exists (spans appear once LLM endpoints are exercised in later tasks).

- [ ] **Step 6: Commit**

```bash
git add backend/app backend/tests/unit/test_middleware.py
git commit -m "feat: phoenix OTLP tracing with OpenInference instrumentation and correlation IDs"
```

---

### Task 9: Hybrid retriever & simple (naive) pipeline + /query API

**Files:**
- Create: `backend/app/retrieval/retriever.py`, `backend/app/pipelines/__init__.py`, `backend/app/pipelines/types.py`, `backend/app/pipelines/simple.py`, `backend/app/api/query.py`
- Modify: `backend/app/main.py` (include query router)
- Test: `backend/tests/unit/test_retriever.py`, `backend/tests/unit/test_simple_pipeline.py`, `backend/tests/unit/test_api_query.py`

**Interfaces:**
- Consumes: Task 5 store factories, Task 7 prompts.
- Produces:
  - `app.pipelines.types.RetrievedChunk` — pydantic: `text: str`, `score: float`, `doc_id: str`, `title: str`, `section_path: str`, `pages: list[int]`.
  - `app.pipelines.types.Citation` — `title: str`, `section_path: str`, `pages: list[int]`.
  - `app.pipelines.types.PipelineResult` — `answer: str`, `citations: list[Citation]`, `chunks: list[RetrievedChunk]`, `grounded: bool | None`, `route: str`, `retrieval_attempts: int`, `generation_attempts: int`.
  - `app.pipelines.types.StatusCallback = Callable[[str], None]`.
  - `app.retrieval.retriever.HybridRetriever(index=None)` — `.retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]` (hybrid PGVector query, deduped by node id); `build_citations(chunks) -> list[Citation]` (module function, unique by (title, section_path)).
  - `app.pipelines.simple.SimplePipeline()` — `.answer(question: str, history: list[dict], on_status: StatusCallback | None = None) -> PipelineResult` (retrieve → synthesize; `grounded=None`, `route="rag"`).
  - `app.api.query` route `POST /api/v1/query` — body `{"question": str, "mode": "agentic"|"simple"|null, "top_k": int|null}`; response `{"answer", "citations", "chunks", "grounded", "mode", "trace_id", "latency_ms"}`. `mode` null → `settings.pipeline_mode`. Used by the RAGAs harness (Task 12).

- [ ] **Step 1: Write failing retriever tests** (fake LlamaIndex retriever)

`backend/tests/unit/test_retriever.py`:

```python
from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode


def _node(nid, text, score, **meta):
    base = {"doc_id": "d1", "title": "Doc", "section_path": "S", "pages": [1], "is_table": False}
    base.update(meta)
    return NodeWithScore(node=TextNode(id_=nid, text=text, metadata=base), score=score)


def test_retrieve_maps_nodes_and_dedupes():
    from app.retrieval.retriever import HybridRetriever
    index = MagicMock()
    index.as_retriever.return_value.retrieve.return_value = [
        _node("a", "alpha", 0.9), _node("a", "alpha", 0.8), _node("b", "beta", 0.7, title="Doc2"),
    ]
    chunks = HybridRetriever(index=index).retrieve("q", top_k=5)
    assert [c.text for c in chunks] == ["alpha", "beta"]
    assert chunks[0].score == 0.9 and chunks[1].title == "Doc2"


def test_build_citations_unique():
    from app.pipelines.types import RetrievedChunk
    from app.retrieval.retriever import build_citations
    chunks = [
        RetrievedChunk(text="x", score=1.0, doc_id="d", title="Doc", section_path="A", pages=[1]),
        RetrievedChunk(text="y", score=0.9, doc_id="d", title="Doc", section_path="A", pages=[2]),
        RetrievedChunk(text="z", score=0.8, doc_id="d", title="Doc", section_path="B", pages=[3]),
    ]
    cites = build_citations(chunks)
    assert [(c.title, c.section_path) for c in cites] == [("Doc", "A"), ("Doc", "B")]
    assert cites[0].pages == [1, 2]
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implement types + retriever**

`backend/app/pipelines/types.py`:

```python
from collections.abc import Callable

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    score: float
    doc_id: str
    title: str
    section_path: str = ""
    pages: list[int] = Field(default_factory=list)


class Citation(BaseModel):
    title: str
    section_path: str = ""
    pages: list[int] = Field(default_factory=list)


class PipelineResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool | None = None
    route: str = "rag"
    retrieval_attempts: int = 0
    generation_attempts: int = 0


StatusCallback = Callable[[str], None]
```

`backend/app/retrieval/retriever.py`:

```python
import logging

from llama_index.core import VectorStoreIndex

from app.core.config import get_settings
from app.pipelines.types import Citation, RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, index: VectorStoreIndex | None = None) -> None:
        if index is None:
            from app.retrieval.vector_store import get_embed_model, get_vector_store
            index = VectorStoreIndex.from_vector_store(
                get_vector_store(), embed_model=get_embed_model()
            )
        self.index = index

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or get_settings().retrieval_top_k
        retriever = self.index.as_retriever(
            similarity_top_k=k,
            sparse_top_k=k,
            vector_store_query_mode="hybrid",
        )
        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []
        for node_with_score in retriever.retrieve(query):
            node = node_with_score.node
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            meta = node.metadata or {}
            chunks.append(RetrievedChunk(
                text=node.get_content(),
                score=float(node_with_score.score or 0.0),
                doc_id=str(meta.get("doc_id", "")),
                title=str(meta.get("title", "")),
                section_path=str(meta.get("section_path", "")),
                pages=list(meta.get("pages", []) or []),
            ))
        logger.info("retrieved %d chunks for query %r", len(chunks), query[:80])
        return chunks


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    grouped: dict[tuple[str, str], Citation] = {}
    for c in chunks:
        key = (c.title, c.section_path)
        if key not in grouped:
            grouped[key] = Citation(title=c.title, section_path=c.section_path, pages=[])
        grouped[key].pages = sorted(set(grouped[key].pages) | set(c.pages))
    return list(grouped.values())
```

- [ ] **Step 4: Run retriever tests — verify PASS**

- [ ] **Step 5: Write failing simple-pipeline test, then implement**

`backend/tests/unit/test_simple_pipeline.py`:

```python
from unittest.mock import MagicMock

from app.pipelines.types import RetrievedChunk


def test_simple_pipeline_answers_with_citations():
    from app.pipelines.simple import SimplePipeline
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(text="[Doc > A]\n\nTotal: 100", score=0.9,
                       doc_id="d", title="Doc", section_path="A", pages=[1]),
    ]
    llm = MagicMock()
    llm.complete.return_value = MagicMock(text="The total is 100 [Doc, A].")
    statuses = []
    result = SimplePipeline(retriever=retriever, llm=llm).answer(
        "what is the total?", history=[], on_status=statuses.append
    )
    assert result.answer.startswith("The total is 100")
    assert result.citations[0].title == "Doc"
    assert result.grounded is None and result.retrieval_attempts == 1
    assert any("Retrieving" in s for s in statuses)
    prompt_sent = llm.complete.call_args.args[0]
    assert "Total: 100" in prompt_sent and "what is the total?" in prompt_sent


def test_simple_pipeline_no_chunks_message():
    from app.pipelines.simple import SimplePipeline
    retriever = MagicMock(); retriever.retrieve.return_value = []
    llm = MagicMock()
    result = SimplePipeline(retriever=retriever, llm=llm).answer("q", history=[])
    assert "couldn't find" in result.answer.lower()
    llm.complete.assert_not_called()
```

`backend/app/pipelines/simple.py`:

```python
import logging

from app.core.config import get_settings
from app.observability.prompts import get_prompt_manager
from app.pipelines.types import PipelineResult, StatusCallback
from app.retrieval.retriever import HybridRetriever, build_citations

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the document knowledge base for that question."
)


def format_context(chunks) -> str:
    return "\n\n---\n\n".join(c.text for c in chunks)


class SimplePipeline:
    """Naive RAG: retrieve -> synthesize. Also the RAGAs baseline."""

    def __init__(self, retriever: HybridRetriever | None = None, llm=None) -> None:
        s = get_settings()
        self.retriever = retriever or HybridRetriever()
        if llm is None:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model=s.llm_model, base_url=s.ollama_base_url,
                         request_timeout=s.llm_timeout_seconds, temperature=0.0)
        self.llm = llm

    def answer(self, question: str, history: list[dict],
               on_status: StatusCallback | None = None) -> PipelineResult:
        notify = on_status or (lambda _msg: None)
        notify("Retrieving documents…")
        chunks = self.retriever.retrieve(question)
        if not chunks:
            return PipelineResult(answer=NO_CONTEXT_ANSWER, retrieval_attempts=1)
        notify("Synthesizing answer…")
        prompt = get_prompt_manager().get(
            "synthesizer", context=format_context(chunks), question=question, feedback=""
        )
        answer = self.llm.complete(prompt).text.strip()
        return PipelineResult(
            answer=answer, citations=build_citations(chunks), chunks=chunks,
            retrieval_attempts=1, generation_attempts=1,
        )
```

Run: `uv run pytest tests/unit/test_simple_pipeline.py -v` — Expected: 2 PASS.

- [ ] **Step 6: Write failing /query API test, then implement**

`backend/tests/unit/test_api_query.py`:

```python
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.pipelines.types import Citation, PipelineResult

AUTH = {"Authorization": "Bearer documind-dev-key"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.db.session as db_session
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    db_session.get_engine.cache_clear()
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


def test_query_simple_mode(client):
    result = PipelineResult(answer="42 [Doc, A]", citations=[Citation(title="Doc")])
    with patch("app.api.query.get_pipeline") as gp:
        gp.return_value.answer = MagicMock(return_value=result)
        r = client.post("/api/v1/query", headers=AUTH,
                        json={"question": "meaning of life?", "mode": "simple"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "42 [Doc, A]"
    assert body["mode"] == "simple"
    assert "latency_ms" in body and "trace_id" in body


def test_query_validates_empty_question(client):
    r = client.post("/api/v1/query", headers=AUTH, json={"question": "  "})
    assert r.status_code == 422
```

`backend/app/api/query.py`:

```python
import time
from typing import Literal

from fastapi import APIRouter
from opentelemetry import trace
from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.pipelines.types import Citation, PipelineResult, RetrievedChunk

router = APIRouter(prefix="/api/v1", tags=["query"])


class QueryIn(BaseModel):
    question: str
    mode: Literal["agentic", "simple"] | None = None
    top_k: int | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]
    grounded: bool | None
    mode: str
    trace_id: str
    latency_ms: int


def get_pipeline(mode: str):
    if mode == "simple":
        from app.pipelines.simple import SimplePipeline
        return SimplePipeline()
    from app.pipelines.agentic import AgenticPipeline
    return AgenticPipeline()


@router.post("/query", response_model=QueryOut)
def query(body: QueryIn) -> QueryOut:
    mode = body.mode or get_settings().pipeline_mode
    start = time.perf_counter()
    result: PipelineResult = get_pipeline(mode).answer(body.question, history=[])
    latency_ms = int((time.perf_counter() - start) * 1000)
    ctx = trace.get_current_span().get_span_context()
    trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else ""
    return QueryOut(
        answer=result.answer, citations=result.citations, chunks=result.chunks,
        grounded=result.grounded, mode=mode, trace_id=trace_id, latency_ms=latency_ms,
    )
```

In `create_app()` include the router (protected): `from app.api import query` … `app.include_router(query.router, dependencies=protected)`.

Note: `AgenticPipeline` doesn't exist until Task 10 — the import is inside `get_pipeline`, so simple-mode tests pass now; the two query tests above only exercise `simple`.

Run: `uv run pytest tests/unit/test_api_query.py -v` — Expected: 2 PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app backend/tests
git commit -m "feat: hybrid retriever, naive pipeline baseline and /query API"
```

---

### Task 10: CrewAI corrective-RAG agentic pipeline

**Files:**
- Create: `backend/app/agents/__init__.py`, `backend/app/agents/llm.py`, `backend/app/agents/tools.py`, `backend/app/agents/stages.py`, `backend/app/pipelines/agentic.py`
- Test: `backend/tests/unit/test_agentic_parsing.py`, `backend/tests/unit/test_agentic_orchestration.py`

**Interfaces:**
- Consumes: Task 7 `get_prompt_manager()`, Task 9 `HybridRetriever`/`build_citations`/`format_context`/`NO_CONTEXT_ANSWER`, types from `app.pipelines.types`.
- Produces:
  - `app.agents.llm.get_crew_llm() -> crewai.LLM` (`model=f"ollama/{settings.llm_model}"`, `base_url=settings.ollama_base_url`, `temperature=0.0`).
  - `app.agents.tools.DocumentSearchTool(retriever, buffer)` — CrewAI `BaseTool` named `document_search`; arg `query: str`; formats results as numbered chunks; appends raw `RetrievedChunk`s to the shared `buffer: list`.
  - `app.agents.stages.CrewStages(llm=None, retriever=None)` — one method per agent role, each building a single-agent CrewAI `Crew` and returning parsed output: `route(question) -> str` (`"rag"|"direct"`, default `rag`), `rewrite(question, history, feedback) -> list[str]`, `research(queries) -> list[RetrievedChunk]` (agent + tool, with direct-retrieval fallback if the agent never called the tool), `grade(question, chunks) -> list[int]`, `synthesize(question, chunks, feedback) -> str`, `check(answer, chunks) -> bool`, `direct_answer(question) -> str`.
  - Parse helpers (module-level in `stages.py`, pure): `parse_route(text) -> str`, `parse_queries(text) -> list[str]`, `parse_indices(text, n_chunks) -> list[int]`, `parse_verdict(text) -> bool`.
  - `app.pipelines.agentic.AgenticPipeline(stages=None)` — `.answer(question, history, on_status=None) -> PipelineResult`; orchestration below.

**Orchestration contract (implement exactly; tests assert it):**
1. `route` → if `direct`: return `PipelineResult(answer=direct_answer(...), route="direct")` — no retrieval.
2. Loop (max `settings.max_retrieval_attempts` = 2): `rewrite` (attempt 2 gets grader feedback) → `research` → `grade`. If graded-relevant chunks non-empty → break.
3. No relevant chunks after loop → return `NO_CONTEXT_ANSWER` result (`grounded=None`).
4. Loop (max `settings.max_generation_attempts` = 2): `synthesize` (attempt 2 gets "previous answer was not grounded" feedback) → `check`. If grounded → break.
5. Return `PipelineResult` with answer, `build_citations(relevant_chunks)`, chunks, `grounded` (last verdict), route, attempt counters. Emit `on_status` messages: "Routing query…", "Rewriting query…", "Searching documents…", "Grading context…", "Synthesizing answer…", "Verifying groundedness…".

- [ ] **Step 1: Write failing parser tests**

`backend/tests/unit/test_agentic_parsing.py`:

```python
from app.agents.stages import parse_indices, parse_queries, parse_route, parse_verdict


def test_parse_route():
    assert parse_route("rag") == "rag"
    assert parse_route(" Direct.\n") == "direct"
    assert parse_route("gibberish") == "rag"  # safe default: retrieve


def test_parse_queries():
    assert parse_queries('["a", "b"]') == ["a", "b"]
    assert parse_queries('Here you go: ["invoice total June"]') == ["invoice total June"]
    assert parse_queries("not json", fallback="orig q") == ["orig q"]


def test_parse_indices():
    assert parse_indices("[0, 2]", n_chunks=3) == [0, 2]
    assert parse_indices("[0, 9]", n_chunks=3) == [0]   # out of range dropped
    assert parse_indices("none relevant []", n_chunks=3) == []
    assert parse_indices("garbage", n_chunks=2) == [0, 1]  # unparseable: keep all


def test_parse_verdict():
    assert parse_verdict("yes") is True
    assert parse_verdict(" No\n") is False
    assert parse_verdict("unclear") is True  # fail open: do not block answers
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implement llm, tool, stages**

`backend/app/agents/llm.py`:

```python
from functools import lru_cache

from crewai import LLM

from app.core.config import get_settings


@lru_cache
def get_crew_llm() -> LLM:
    s = get_settings()
    return LLM(model=f"ollama/{s.llm_model}", base_url=s.ollama_base_url,
               temperature=0.0, timeout=s.llm_timeout_seconds)
```

`backend/app/agents/tools.py`:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from app.pipelines.types import RetrievedChunk
from app.retrieval.retriever import HybridRetriever


class SearchInput(BaseModel):
    query: str = Field(description="standalone search query for the document knowledge base")


class DocumentSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "document_search"
    description: str = "Search the PDF knowledge base. Returns numbered text chunks."
    args_schema: type[BaseModel] = SearchInput
    retriever: HybridRetriever
    buffer: list[RetrievedChunk]

    def _run(self, query: str) -> str:
        chunks = self.retriever.retrieve(query)
        self.buffer.extend(chunks)
        if not chunks:
            return "No results found."
        return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
```

`backend/app/agents/stages.py`:

```python
import json
import logging
import re

from crewai import Agent, Crew, Process, Task

from app.agents.llm import get_crew_llm
from app.agents.tools import DocumentSearchTool
from app.observability.prompts import get_prompt_manager
from app.pipelines.simple import format_context
from app.pipelines.types import RetrievedChunk
from app.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


# --- pure parse helpers (tested directly) ---

def parse_route(text: str) -> str:
    return "direct" if "direct" in text.strip().lower() else "rag"


def _extract_json_array(text: str) -> list | None:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def parse_queries(text: str, fallback: str = "") -> list[str]:
    arr = _extract_json_array(text)
    queries = [q.strip() for q in (arr or []) if isinstance(q, str) and q.strip()]
    return queries[:3] if queries else ([fallback] if fallback else [])


def parse_indices(text: str, n_chunks: int) -> list[int]:
    arr = _extract_json_array(text)
    if arr is None:
        return list(range(n_chunks))  # unparseable -> keep everything
    return sorted({int(i) for i in arr if isinstance(i, (int, float)) and 0 <= int(i) < n_chunks})


def parse_verdict(text: str) -> bool:
    return not text.strip().lower().startswith("no")


# --- crew stages: one single-agent Crew per pipeline step ---

class CrewStages:
    def __init__(self, llm=None, retriever: HybridRetriever | None = None) -> None:
        self.llm = llm or get_crew_llm()
        self.retriever = retriever or HybridRetriever()
        self.prompts = get_prompt_manager()

    def _kickoff(self, role: str, goal: str, description: str,
                 expected_output: str, tools: list | None = None) -> str:
        agent = Agent(role=role, goal=goal, backstory="You work inside DocuMind, a document search platform.",
                      llm=self.llm, tools=tools or [], verbose=False, allow_delegation=False, max_iter=3)
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return str(crew.kickoff())

    def route(self, question: str) -> str:
        out = self._kickoff("Query Router", "Classify queries precisely",
                            self.prompts.get("router", question=question), "one word: rag or direct")
        return parse_route(out)

    def rewrite(self, question: str, history: str, feedback: str) -> list[str]:
        fb = f"\nFeedback from a failed retrieval attempt: {feedback}" if feedback else ""
        out = self._kickoff("Query Rewriter", "Produce excellent standalone search queries",
                            self.prompts.get("rewriter", question=question, history=history or "(empty)", feedback=fb),
                            "JSON array of 1-3 query strings")
        return parse_queries(out, fallback=question)

    def research(self, queries: list[str]) -> list[RetrievedChunk]:
        buffer: list[RetrievedChunk] = []
        tool = DocumentSearchTool(retriever=self.retriever, buffer=buffer)
        description = (
            "Use the document_search tool once per query to gather evidence. Queries:\n"
            + "\n".join(f"- {q}" for q in queries)
            + "\nAfter searching, reply with a one-line summary of what was found."
        )
        try:
            self._kickoff("Researcher", "Gather relevant document evidence",
                          description, "one-line summary", tools=[tool])
        except Exception as exc:
            logger.warning("researcher crew failed (%s); falling back to direct retrieval", exc)
        if not buffer:  # agent never called the tool -> deterministic fallback
            for q in queries:
                buffer.extend(self.retriever.retrieve(q))
        seen: set[str] = set()
        unique = [c for c in buffer if not (c.text in seen or seen.add(c.text))]
        return unique

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> list[int]:
        numbered = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
        out = self._kickoff("Relevance Grader", "Judge context strictly",
                            self.prompts.get("grader", question=question, chunks=numbered),
                            "JSON array of relevant chunk numbers")
        return parse_indices(out, n_chunks=len(chunks))

    def synthesize(self, question: str, chunks: list[RetrievedChunk], feedback: str) -> str:
        fb = f"\nIMPORTANT: {feedback}" if feedback else ""
        return self._kickoff("Answer Synthesizer", "Write grounded, cited answers",
                             self.prompts.get("synthesizer", context=format_context(chunks),
                                              question=question, feedback=fb),
                             "a grounded answer with [title, section] citations").strip()

    def check(self, answer: str, chunks: list[RetrievedChunk]) -> bool:
        out = self._kickoff("Groundedness Checker", "Detect unsupported claims",
                            self.prompts.get("hallucination_checker",
                                             context=format_context(chunks), answer=answer),
                            "one word: yes or no")
        return parse_verdict(out)

    def direct_answer(self, question: str) -> str:
        return self._kickoff(
            "Assistant", "Answer briefly and helpfully",
            f"You are DocuMind, a document search assistant. Reply briefly to: {question}",
            "a short friendly reply").strip()
```

`backend/app/pipelines/agentic.py`:

```python
import logging

from app.core.config import get_settings
from app.pipelines.simple import NO_CONTEXT_ANSWER
from app.pipelines.types import PipelineResult, StatusCallback
from app.retrieval.retriever import build_citations

logger = logging.getLogger(__name__)


class AgenticPipeline:
    """Corrective RAG: route -> (rewrite -> research -> grade)* -> (synthesize -> check)*."""

    def __init__(self, stages=None) -> None:
        if stages is None:
            from app.agents.stages import CrewStages
            stages = CrewStages()
        self.stages = stages
        self.settings = get_settings()

    def answer(self, question: str, history: list[dict],
               on_status: StatusCallback | None = None) -> PipelineResult:
        notify = on_status or (lambda _msg: None)
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

        notify("Routing query…")
        if self.stages.route(question) == "direct":
            return PipelineResult(answer=self.stages.direct_answer(question), route="direct")

        relevant, all_chunks, feedback = [], [], ""
        attempts = 0
        for attempt in range(self.settings.max_retrieval_attempts):
            attempts = attempt + 1
            notify("Rewriting query…")
            queries = self.stages.rewrite(question, history_text, feedback)
            notify("Searching documents…")
            all_chunks = self.stages.research(queries)
            if not all_chunks:
                feedback = "No documents matched; try broader or different terms."
                continue
            notify("Grading context…")
            keep = self.stages.grade(question, all_chunks)
            relevant = [all_chunks[i] for i in keep]
            if relevant:
                break
            feedback = "Retrieved chunks were judged irrelevant; rephrase with more specific terms."

        if not relevant:
            return PipelineResult(answer=NO_CONTEXT_ANSWER, retrieval_attempts=attempts)

        grounded, answer, gen_attempts = True, "", 0
        feedback = ""
        for attempt in range(self.settings.max_generation_attempts):
            gen_attempts = attempt + 1
            notify("Synthesizing answer…")
            answer = self.stages.synthesize(question, relevant, feedback)
            notify("Verifying groundedness…")
            grounded = self.stages.check(answer, relevant)
            if grounded:
                break
            feedback = ("Your previous answer contained claims not supported by the context. "
                        "Use only facts from the context.")

        return PipelineResult(
            answer=answer, citations=build_citations(relevant), chunks=relevant,
            grounded=grounded, route="rag",
            retrieval_attempts=attempts, generation_attempts=gen_attempts,
        )
```

Run parser tests: `uv run pytest tests/unit/test_agentic_parsing.py -v` — Expected: 4 PASS.

- [ ] **Step 4: Write failing orchestration tests, verify they pass against the code above**

`backend/tests/unit/test_agentic_orchestration.py`:

```python
from unittest.mock import MagicMock

from app.pipelines.types import RetrievedChunk


def _chunk(text="c1"):
    return RetrievedChunk(text=text, score=0.9, doc_id="d", title="Doc", section_path="S", pages=[1])


def _stages(route="rag", grades=None, verdicts=None, chunks=None):
    s = MagicMock()
    s.route.return_value = route
    s.rewrite.return_value = ["q1"]
    s.research.return_value = chunks if chunks is not None else [_chunk()]
    s.grade.side_effect = grades or [[0]]
    s.check.side_effect = verdicts or [True]
    s.synthesize.return_value = "answer [Doc, S]"
    s.direct_answer.return_value = "hi!"
    return s


def _pipeline(stages):
    from app.pipelines.agentic import AgenticPipeline
    return AgenticPipeline(stages=stages)


def test_direct_route_skips_retrieval():
    stages = _stages(route="direct")
    result = _pipeline(stages).answer("hello", history=[])
    assert result.route == "direct" and result.answer == "hi!"
    stages.rewrite.assert_not_called()


def test_happy_path_single_pass():
    stages = _stages()
    result = _pipeline(stages).answer("q", history=[])
    assert result.answer.startswith("answer")
    assert result.grounded is True
    assert result.retrieval_attempts == 1 and result.generation_attempts == 1
    assert result.citations and result.citations[0].title == "Doc"


def test_corrective_loop_bounded_to_two_attempts():
    stages = _stages(grades=[[], []])  # grader rejects everything, twice
    result = _pipeline(stages).answer("q", history=[])
    assert stages.rewrite.call_count == 2          # exactly one retry
    assert "couldn't find" in result.answer.lower()
    assert result.retrieval_attempts == 2
    # second rewrite got corrective feedback
    assert stages.rewrite.call_args_list[1].args[2] != ""


def test_regeneration_on_hallucination_bounded():
    stages = _stages(verdicts=[False, False])
    result = _pipeline(stages).answer("q", history=[])
    assert stages.synthesize.call_count == 2       # exactly one regeneration
    assert result.grounded is False                # shipped with honest flag
    assert result.generation_attempts == 2
```

Run: `uv run pytest tests/unit/test_agentic_orchestration.py -v` — Expected: 4 PASS. (If any fail, fix `agentic.py` — the contract above is authoritative.)

- [ ] **Step 5: Live smoke test** — stack up + documents ingested:
`curl -s -X POST localhost:8000/api/v1/query -H "Authorization: Bearer documind-dev-key" -H "Content-Type: application/json" -d "{\"question\": \"What is the invoice total?\", \"mode\": \"agentic\"}"`
Expected: JSON answer with citations in 1–3 min (CPU); Phoenix (`localhost:6006`, project `documind`) shows a nested trace: crew kickoffs per stage with LLM spans inside.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agents backend/app/pipelines/agentic.py backend/tests
git commit -m "feat: crewai corrective-RAG pipeline with bounded retry loops"
```

---

### Task 11: OpenAI-compatible chat API with streaming

**Files:**
- Create: `backend/app/api/openai_compat.py`
- Modify: `backend/app/main.py` (include router with auth)
- Test: `backend/tests/unit/test_openai_compat.py`

**Interfaces:**
- Consumes: `get_pipeline(mode)` from Task 9, `PipelineResult`.
- Produces:
  - `GET /v1/models` → `{"object": "list", "data": [{"id": "agentic-rag", "object": "model", "owned_by": "documind"}]}`.
  - `POST /v1/chat/completions` — accepts standard OpenAI body (`model`, `messages`, `stream`). Non-stream → standard `chat.completion` object. Stream → SSE `chat.completion.chunk` frames ending with `data: [DONE]`. Status updates stream inside a `<think>…</think>` block (OpenWebUI renders it as a collapsible "thinking" panel), then the answer streams in ~48-char pieces, then a final source-list line.
  - Helper functions (unit-tested): `extract_question_and_history(messages) -> tuple[str, list[dict]]` (last user message = question; earlier user/assistant messages = history with `<think>` blocks stripped), `format_sources(result) -> str` (returns "" when no citations, else `\n\n**Sources:** …` line).

- [ ] **Step 1: Write failing tests**

`backend/tests/unit/test_openai_compat.py`:

```python
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.pipelines.types import Citation, PipelineResult

AUTH = {"Authorization": "Bearer documind-dev-key"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.db.session as db_session
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    db_session.get_engine.cache_clear()
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


def _result():
    return PipelineResult(answer="Total is 100.", grounded=True,
                          citations=[Citation(title="Invoice", section_path="Summary", pages=[1])])


def test_models_endpoint(client):
    r = client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "agentic-rag"


def test_extract_question_and_history():
    from app.api.openai_compat import extract_question_and_history
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "<think>hmm</think>first a"},
        {"role": "user", "content": "second q"},
    ]
    q, hist = extract_question_and_history(messages)
    assert q == "second q"
    assert hist == [{"role": "user", "content": "first q"},
                    {"role": "assistant", "content": "first a"}]


def test_chat_completion_non_streaming(client):
    with patch("app.api.openai_compat.get_pipeline") as gp:
        gp.return_value.answer = MagicMock(return_value=_result())
        r = client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "agentic-rag", "stream": False,
            "messages": [{"role": "user", "content": "total?"}],
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert "Total is 100." in content and "**Sources:**" in content
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming(client):
    with patch("app.api.openai_compat.get_pipeline") as gp:
        def fake_answer(question, history, on_status=None):
            if on_status:
                on_status("Routing query…")
            return _result()
        gp.return_value.answer = fake_answer
        r = client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "agentic-rag", "stream": True,
            "messages": [{"role": "user", "content": "total?"}],
        })
    assert r.status_code == 200
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(l[6:]) for l in lines[:-1]]
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    full = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert "<think>" in full and "Routing query…" in full and "</think>" in full
    assert "Total is 100." in full
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_empty_messages_rejected(client):
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "agentic-rag", "messages": []})
    assert r.status_code == 422
```

- [ ] **Step 2: Run — verify FAIL**

- [ ] **Step 3: Implement**

`backend/app/api/openai_compat.py`:

```python
import asyncio
import json
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.query import get_pipeline
from app.core.config import get_settings
from app.pipelines.types import PipelineResult

router = APIRouter(prefix="/v1", tags=["openai-compat"])

MODEL_ID = "agentic-rag"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


@router.get("/models")
def list_models() -> dict:
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "documind"}]}


def extract_question_and_history(messages: list[dict | ChatMessage]) -> tuple[str, list[dict]]:
    msgs = [m if isinstance(m, dict) else m.model_dump() for m in messages]
    msgs = [m for m in msgs if m["role"] in ("user", "assistant")]
    question = ""
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            question = msgs[i]["content"]
            msgs = msgs[:i]
            break
    history = [{"role": m["role"], "content": _THINK_RE.sub("", m["content"]).strip()}
               for m in msgs]
    return question.strip(), history


def format_sources(result: PipelineResult) -> str:
    if not result.citations:
        return ""
    parts = []
    for c in result.citations:
        label = f"{c.title} > {c.section_path}" if c.section_path else c.title
        if c.pages:
            label += f" (p. {', '.join(map(str, c.pages))})"
        parts.append(label)
    note = "" if result.grounded in (True, None) else " ⚠️ groundedness check did not pass"
    return "\n\n**Sources:** " + " · ".join(parts) + note


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chunk_frame(cid: str, created: int, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    question, history = extract_question_and_history(body.messages)
    if not question:
        raise HTTPException(422, "no user message found")
    mode = get_settings().pipeline_mode
    pipeline = get_pipeline(mode)

    if not body.stream:
        result = await asyncio.to_thread(pipeline.answer, question, history)
        content = result.answer + format_sources(result)
        return {
            "id": _completion_id(), "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    cid, created = _completion_id(), int(time.time())
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_status(msg: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    async def generate():
        yield _chunk_frame(cid, created, {"role": "assistant", "content": "<think>\n"})
        task = loop.run_in_executor(None, lambda: pipeline.answer(question, history, on_status))
        while True:
            get_status = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({get_status, task}, return_when=asyncio.FIRST_COMPLETED)
            if get_status in done:
                yield _chunk_frame(cid, created, {"content": f"{get_status.result()}\n"})
                continue
            get_status.cancel()
            break
        result: PipelineResult = task.result()
        while not queue.empty():  # drain late statuses
            yield _chunk_frame(cid, created, {"content": f"{queue.get_nowait()}\n"})
        yield _chunk_frame(cid, created, {"content": "</think>\n\n"})
        text = result.answer + format_sources(result)
        for i in range(0, len(text), 48):
            yield _chunk_frame(cid, created, {"content": text[i:i + 48]})
        yield _chunk_frame(cid, created, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

In `create_app()`: `from app.api import openai_compat` … `app.include_router(openai_compat.router, dependencies=protected)`.

- [ ] **Step 4: Run — verify PASS** (`uv run pytest tests/unit/test_openai_compat.py -v` → 5 PASS; full suite green)

- [ ] **Step 5: Live OpenWebUI check** — `docker compose up -d --build backend openwebui`; open `http://localhost:3000`; model `agentic-rag` appears in the dropdown; ask "What documents do you have about invoices?" → collapsible thinking panel with stage statuses, then the cited answer. Trace visible in Phoenix.

- [ ] **Step 6: Commit**

```bash
git add backend/app backend/tests
git commit -m "feat: OpenAI-compatible chat API with think-block status streaming"
```

---

### Task 12: RAGAs evaluation harness & golden dataset

**Files:**
- Create: `evaluation/golden_set.json`, `scripts/evaluate.py`
- Test: manual run (offline tooling; the harness is not unit-tested — its output artifacts are reviewed instead)

**Interfaces:**
- Consumes: running backend (`POST /api/v1/query` with `mode`), Ollama judge model.
- Produces: `doc/evaluation-report.md` + `evaluation/runs/<timestamp>/results.json` (gitignored raw runs).

- [ ] **Step 1: Author the golden dataset — requires reading the local PDFs**

Open each PDF in `data/documents/` and write **15+ entries** in `evaluation/golden_set.json`. Every `reference` answer must be verbatim-verifiable from a document. Include ≥2 unanswerable and ≥1 multi-document questions. Structure (first entries shown as authoring examples — replace values with facts from the real files):

```json
[
  {
    "question": "What is the total amount on the June 2026 invoice?",
    "reference": "The total amount on the June 2026 invoice is <actual amount from the PDF>.",
    "category": "single-doc-numeric"
  },
  {
    "question": "What is the net pay in the June 2026 payslip?",
    "reference": "The net pay for June 2026 is <actual amount>.",
    "category": "single-doc-numeric"
  },
  {
    "question": "Compare the two June 2026 invoices: which one has the higher total?",
    "reference": "<which invoice> has the higher total (<amount A> vs <amount B>).",
    "category": "multi-doc"
  },
  {
    "question": "What is the company's stock price today?",
    "reference": "The knowledge base does not contain stock price information.",
    "category": "unanswerable"
  }
]
```

- [ ] **Step 2: Write the evaluation script**

`scripts/evaluate.py` (run: `uv run --project backend --group eval python scripts/evaluate.py --modes simple agentic`):

```python
"""RAGAs evaluation: agentic vs simple pipeline over the golden set."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "http://localhost:8000"


def run_pipeline(question: str, mode: str, api_key: str) -> dict:
    r = httpx.post(
        f"{BACKEND}/api/v1/query",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"question": question, "mode": mode},
        timeout=600.0,
    )
    r.raise_for_status()
    return r.json()


def collect(golden: list[dict], mode: str, api_key: str) -> list[dict]:
    rows = []
    for i, item in enumerate(golden):
        start = time.perf_counter()
        out = run_pipeline(item["question"], mode, api_key)
        latency = time.perf_counter() - start
        rows.append({
            "user_input": item["question"],
            "response": out["answer"],
            "retrieved_contexts": [c["text"] for c in out["chunks"]],
            "reference": item["reference"],
            "category": item.get("category", ""),
            "latency_s": round(latency, 2),
            "grounded": out.get("grounded"),
            "trace_id": out.get("trace_id", ""),
        })
        print(f"[{mode}] {i + 1}/{len(golden)} {latency:.1f}s")
    return rows


def ragas_scores(rows: list[dict], judge_model: str, embed_model: str, ollama_url: str) -> dict:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    judge = LangchainLLMWrapper(ChatOllama(model=judge_model, base_url=ollama_url, temperature=0))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embed_model, base_url=ollama_url))
    dataset = EvaluationDataset.from_list(
        [{k: r[k] for k in ("user_input", "response", "retrieved_contexts", "reference")}
         for r in rows]
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                        context_precision, context_recall],
                      llm=judge, embeddings=emb)
    df = result.to_pandas()
    return {m: round(float(df[m].mean()), 3)
            for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}


def latency_stats(rows: list[dict]) -> dict:
    xs = sorted(r["latency_s"] for r in rows)
    return {"p50_s": round(statistics.median(xs), 1),
            "p95_s": round(xs[max(0, int(len(xs) * 0.95) - 1)], 1),
            "mean_s": round(statistics.mean(xs), 1)}


def write_report(all_results: dict, out_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# DocuMind RAG Evaluation Report", "",
        f"Generated: {ts} · Judge: local Ollama (see caveat) · Metrics: RAGAs", "",
        "| Metric | " + " | ".join(all_results) + " |",
        "|---|" + "---|" * len(all_results),
    ]
    metrics = list(next(iter(all_results.values()))["scores"])
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(str(v["scores"][m]) for v in all_results.values()) + " |")
    lines += ["", "## Latency", "",
              "| Stat | " + " | ".join(all_results) + " |",
              "|---|" + "---|" * len(all_results)]
    for stat in ("p50_s", "p95_s", "mean_s"):
        lines.append(f"| {stat} | " + " | ".join(str(v["latency"][stat]) for v in all_results.values()) + " |")
    lines += ["", "## Caveats", "",
              "- Judge is a local 7B model on CPU: scores are directionally useful, noisy in absolute terms.",
              "- Agentic vs simple comparison shares the same judge, so relative differences are more reliable.",
              "- Unanswerable questions score via faithfulness (refusing is grounded behavior)."]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["simple", "agentic"])
    parser.add_argument("--api-key", default="documind-dev-key")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    golden = json.loads((ROOT / "evaluation" / "golden_set.json").read_text(encoding="utf-8"))
    run_dir = ROOT / "evaluation" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    all_results = {}
    for mode in args.modes:
        rows = collect(golden, mode, args.api_key)
        scores = ragas_scores(rows, args.judge_model, args.embed_model, args.ollama_url)
        all_results[mode] = {"scores": scores, "latency": latency_stats(rows), "rows": rows}
        print(f"[{mode}] scores: {scores}")

    (run_dir / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    write_report(all_results, ROOT / "doc" / "evaluation-report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the evaluation** (stack up, documents ingested; expect ~1–2 h wall clock on CPU for 15 questions × 2 modes + judging — leave it running):
`uv run --project backend --group eval python scripts/evaluate.py`
Expected: per-question progress, two score dicts, `doc/evaluation-report.md` written, raw rows under `evaluation/runs/` (gitignored).

- [ ] **Step 4: Sanity-review the report** — metrics in [0,1]; agentic faithfulness ≥ simple faithfulness expected (hallucination checker); if a metric is NaN (judge failed to parse), re-run that mode once; note surviving anomalies in the report's Caveats.

- [ ] **Step 5: Commit**

```bash
git add evaluation/golden_set.json scripts/evaluate.py doc/evaluation-report.md
git commit -m "feat: RAGAs evaluation harness with agentic-vs-naive comparison report"
```

---

### Task 13: Documentation — README, architecture, API docs, ADRs

**Files:**
- Create: `README.md`, `doc/architecture.md`, `doc/api.md`, `doc/openapi.json`, `doc/design-decisions.md`, `scripts/export_openapi.py`

**Interfaces:**
- Consumes: everything built; `app.main.app` for OpenAPI export.

- [ ] **Step 1: OpenAPI export script**

`scripts/export_openapi.py` (run: `uv run --project backend python scripts/export_openapi.py`):

```python
"""Export the FastAPI OpenAPI schema to doc/openapi.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402

out = ROOT / "doc" / "openapi.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print(f"wrote {out}")
```

Run it; commit output.

- [ ] **Step 2: Write README.md** with exactly these sections (each complete, no stubs):
  1. **DocuMind** — one-paragraph pitch + feature bullets (agentic corrective RAG, hybrid search, tracing, PromptOps, evaluation).
  2. **Architecture at a glance** — the service table from the spec + a Mermaid `graph LR` of OpenWebUI → backend → (CrewAI stages) → PGVector/Ollama, with Phoenix tapping all calls; link to `doc/architecture.md` for detail.
  3. **Prerequisites** — Docker Desktop, ~10 GB disk for models/images, no GPU needed.
  4. **Quick start** — numbered: `cp .env.example .env` (set `DOCUMIND_API_KEY`) → `docker compose up -d --build` → wait for `ollama-init` (`docker compose logs -f ollama-init`) → drop PDFs into `data/documents/` → `curl -X POST localhost:8000/api/v1/ingest -H "Authorization: Bearer <key>"` → open `http://localhost:3000`, pick `agentic-rag`, chat. Include the expected-latency note (CPU, ~5-8 LLM calls/answer) and the `DOCUMIND_PIPELINE_MODE=simple` fast mode.
  5. **Configuration** — table of every `DOCUMIND_*` variable: name, default, purpose.
  6. **REST API** — summary table of all endpoints; links to Swagger UI (`localhost:8000/docs`), `doc/api.md`, `doc/openapi.json`.
  7. **Observability & PromptOps** — Phoenix UI (`localhost:6006`), what a trace shows, how to edit prompts in Phoenix vs YAML.
  8. **Evaluation** — how to run `scripts/evaluate.py`, link to `doc/evaluation-report.md`, judge caveat.
  9. **Development** — local venv setup (`uv sync --all-groups`), running tests (`uv run pytest -m "not integration"`, `RUN_INTEGRATION=1 ...`), repo layout tree.
  10. **Documentation index** — bullets linking every file in `doc/` (per assessment requirement).

- [ ] **Step 3: Write doc/architecture.md** — sections: Context (C4-style Mermaid `graph TB`: user, OpenWebUI, backend, Postgres, Ollama, Phoenix); Ingestion flow (Mermaid `flowchart LR`: discover → sha → docling → chunk → contextualize → embed → index, with ledger side-writes); Agentic pipeline (Mermaid `stateDiagram-v2` of the corrective loops with their bounds); Data model (documents, ingest_jobs, data_rag_chunks tables with columns); Key design properties (idempotency, per-doc failure isolation, bounded loops, prompt versioning, trace propagation). Every diagram must reflect the implemented code, not aspiration.

- [ ] **Step 4: Write doc/api.md** — for each endpoint: method, path, auth, request/response JSON examples (copy real shapes from the schemas), error codes (401/404/422/400). Note that `openapi.json` is authoritative.

- [ ] **Step 5: Write doc/design-decisions.md** — ADR-style entries (Context/Decision/Consequences), one each for: OpenAI-compat integration (vs Pipe/Pipelines); full corrective crew (vs lean crew — user's explicit choice, latency trade-off); ingest-time contextualization (vs query-time LLM enrichment); hybrid search (vs pure vector); single-agent-crew-per-stage orchestration (vs one big sequential crew — testability, bounded loops); YAML+Phoenix prompt strategy; local 7B judge (vs cloud judge); think-block status streaming (vs silent wait).

- [ ] **Step 6: Commit**

```bash
git add README.md doc scripts/export_openapi.py
git commit -m "docs: README, architecture, API reference, ADRs, OpenAPI export"
```

---

### Task 14: Presentation deck & final verification

**Files:**
- Create: `doc/presentation/deck.md` (Marp), `doc/presentation/README.md` (how to render)

- [ ] **Step 1: Write the Marp deck** — `doc/presentation/deck.md` with `marp: true` front-matter, ~12 slides: (1) DocuMind title + one-liner; (2) Problem & requirements; (3) Architecture diagram (reuse Mermaid or ASCII); (4) Mandated-stack mapping table (tool → where it lives in the repo); (5) Ingestion pipeline; (6) Contextual chunking example (before/after header); (7) Corrective-RAG crew diagram with loop bounds; (8) OpenWebUI integration + streaming UX (think-block screenshot placeholder → replace with real screenshot); (9) Observability: Phoenix trace screenshot; (10) PromptOps flow; (11) Evaluation results table (from `doc/evaluation-report.md`) + caveats; (12) Trade-offs & future work (GPU models, reranking, semantic caching, multi-tenant). `doc/presentation/README.md`: render with `npx @marp-team/marp-cli deck.md -o deck.pdf` (or present the .md directly); note where to drop the two screenshots.

- [ ] **Step 2: Capture screenshots** — OpenWebUI chat with expanded think-block + cited answer; Phoenix trace tree of one agentic query. Save as `doc/presentation/img/chat.png`, `doc/presentation/img/trace.png`; reference them in the deck.

- [ ] **Step 3: Full verification pass (superpowers:verification-before-completion)** — from a clean state:

```bash
docker compose down && docker compose up -d --build
# wait for healthy; then:
curl -s localhost:8000/health                                   # all deps true
curl -s -X POST localhost:8000/api/v1/ingest -H "Authorization: Bearer <key>"   # 202, poll job to completed
curl -s localhost:8000/v1/models -H "Authorization: Bearer <key>"               # agentic-rag listed
# OpenWebUI chat at :3000 answers with citations; Phoenix :6006 shows the trace
cd backend && uv run pytest -m "not integration" -v             # all green
RUN_INTEGRATION=1 uv run pytest -v                              # all green
```

Record actual outputs; fix anything red before proceeding.

- [ ] **Step 4: Repo hygiene & push**

```bash
git status              # confirm: no data/, no .env, no evaluation/runs tracked
git log --oneline       # sensible history
# create GitHub repo "documind" (default branch master) and push:
git remote add origin <github-url>
git push -u origin master
```

- [ ] **Step 5: Commit any final fixes and confirm README quick-start works verbatim on a fresh clone** (clone to a temp dir, follow steps 1-6 of Quick start exactly, chat once).

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** ingestion (T4-T6), contextual agentic RAG (T9-T10), tracing (T8), externalized prompts/PromptOps (T7), evaluation (T12), OpenAI-compat + OpenWebUI (T2, T11), REST + Swagger (T6, T9, T13), deliverable docs/deck (T13-T14). Spec §4.2's "trace_id in /query response" → T9. Hybrid search → T5/T9.
- **Type consistency:** `PipelineResult`/`RetrievedChunk`/`Citation` defined once in T9 `app/pipelines/types.py`; T10/T11/T12 consume them. `get_pipeline` defined in T9, reused in T11. `NO_CONTEXT_ANSWER`/`format_context` defined in T9 `simple.py`, consumed in T10.
- **Known API-drift hotspots** (flagged inline): Phoenix client prompts API (T7), `phoenix.otel.register` signature (T8), CrewAI `BaseTool`/`LLM` import paths (T10), RAGAs metric imports (T12). The Global Constraints rule covers adaptation.
