"""Synchronous single-shot query endpoint.

Dispatches to whichever pipeline (`simple` naive baseline or `agentic`
self-correcting graph) `mode` selects, so it can serve both live product
queries and the RAGAs evaluation harness (Task 12) against the same
contract.
"""

import logging
import time
from typing import Literal

from fastapi import APIRouter
from opentelemetry import trace
from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.pipelines.types import Citation, PipelineResult, RetrievedChunk

logger = logging.getLogger(__name__)
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
    # Imported lazily: AgenticPipeline does not exist until Task 10, and
    # deferring the import means simple-mode requests (and this module's
    # unit tests) work fine without it.
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
    logger.info(
        "query answered mode=%s latency_ms=%d chunks=%d trace_id=%s",
        mode, latency_ms, len(result.chunks), trace_id,
    )
    return QueryOut(
        answer=result.answer,
        citations=result.citations,
        chunks=result.chunks,
        grounded=result.grounded,
        mode=mode,
        trace_id=trace_id,
        latency_ms=latency_ms,
    )
