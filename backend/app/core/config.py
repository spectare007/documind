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
