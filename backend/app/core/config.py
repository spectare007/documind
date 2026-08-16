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

    # Chat streaming (Task 11 fix round 1): bounded, dedicated thread pool
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
