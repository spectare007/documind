"""Shared pipeline result types.

Consumed verbatim by the agentic pipeline (Task 10), the streaming/chat layer
(Task 11) and the RAGAs evaluation harness (Task 12) -- field names and types
here are a stable contract; do not rename or add required fields without
updating all three.
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
    # True when the relevance grader rejected every retrieved chunk but the
    # pipeline still proceeded to synthesis using the top-scored retrieved
    # chunks instead of refusing. See `app.pipelines.agentic` for the
    # rationale. Additive/defaulted -- Task 11 and Task 12 construct
    # `PipelineResult` without this field and must keep working.
    grader_fallback: bool = False


StatusCallback = Callable[[str], None]
