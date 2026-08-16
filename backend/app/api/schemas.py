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
