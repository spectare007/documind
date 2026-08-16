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
    # Content-capture switch for tracing (fix for a review finding: the
    # corpus here is real personal financial documents -- payslips, invoices,
    # a filed tax form -- and Phoenix's `phoenix_data` volume had no
    # retention policy and no way to turn content capture off). True keeps
    # today's behaviour unchanged (prompts, retrieved chunk text and
    # completions are exported to Phoenix, as they always have been). Set to
    # False before pointing this at a real corpus on a shared machine: span
    # structure, timings and token counts are still exported, only the
    # prompt/completion/chunk *text* is redacted. See
    # `app.observability.tracing._trace_config` for how this is wired into
    # OpenInference's own masking config, and the README's "Data handling and
    # security posture" section for the full story.
    trace_content: bool = True
    # Maximum accepted size, in bytes, for a single `POST /api/v1/documents`
    # upload (fix for a review finding: the endpoint used to read an
    # unbounded file fully into memory before writing it). 25 MiB comfortably
    # covers a scanned multi-page payslip/invoice/tax filing while bounding
    # both memory and disk exposure from an unauthenticated-content upload.
    max_upload_bytes: int = 25 * 1024 * 1024

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
    # Default is `simple` on measured behaviour, not preference: against the
    # real corpus agentic mode answered 15 of 23 answerable golden-set
    # questions at a median of 125s, while simple mode answered correctly
    # every question it was tried against, in 25 to 82s, from the same index
    # (doc/evaluation-report.md). Agentic mode stays fully supported and is
    # opt-in, per request (`"mode": "agentic"`) or per deployment
    # (`DOCUMIND_PIPELINE_MODE=agentic`).
    pipeline_mode: Literal["agentic", "simple"] = "simple"
    retrieval_top_k: int = 6
    max_retrieval_attempts: int = 2
    max_generation_attempts: int = 2
    # Whole-request wall-clock budget for the agentic pipeline.
    # `llm_timeout_seconds` bounds a *single* completion, but one agentic
    # request makes up to 11 of them (router, 2x rewriter, 2x researcher,
    # up to 6 grader verdicts, 2x synthesizer, 2x checker), so a degraded
    # Ollama could hold a worker for ~25 minutes without this. Checked at
    # stage boundaries: it stops the pipeline starting *more* work and
    # returns the best result it already has -- it cannot interrupt a
    # completion already in flight, which is what `llm_timeout_seconds`
    # is for.
    request_budget_seconds: float = 300.0

    # Chat streaming: bounded, dedicated thread pool
    # for agentic pipeline runs kicked off by the streaming
    # /v1/chat/completions endpoint (see
    # app.api.openai_compat._get_stream_executor). Kept separate from
    # asyncio's shared default executor -- which asyncio.to_thread() and
    # /api/v1/query's non-streaming call also use -- so that streams
    # abandoned by a disconnecting client (which cannot be interrupted
    # mid-LLM-call, so keep occupying a worker thread until they finish or
    # request_budget_seconds gives up on them) can only ever starve this one
    # endpoint under repeated aborts, never the rest of the app.
    chat_stream_max_workers: int = 4

    # Ingestion
    data_dir: Path = Path("data/documents")
    prompts_dir: Path = Path("prompts")
    chunk_max_tokens: int = 512

    # Vector store
    vector_table_name: str = "rag_chunks"


@lru_cache
def get_settings() -> Settings:
    return Settings()
