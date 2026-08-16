"""Shared pipeline result types.

Consumed verbatim by the agentic pipeline (`app.pipelines.agentic`), the
streaming/chat layer (`app.api.openai_compat`) and the RAGAs evaluation
harness -- field names and types here are a stable contract; do not rename
or add required fields without updating all three.
"""

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
