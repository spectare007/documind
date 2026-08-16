"""Synchronous single-shot query endpoint.

Dispatches to whichever pipeline (`simple` naive baseline or `agentic`
self-correcting graph) `mode` selects, so it can serve both live product
queries and the RAGAs evaluation harness against the same
contract.

--- LLM-unavailable handling (fix for a review finding) ---

Neither pipeline wraps its own LLM call, so a connection failure or a
`llm_timeout_seconds` timeout used to propagate out of this handler as
FastAPI's generic unhandled 500 -- no structured body, no route-level log.
That matters here specifically because the RAGAs evaluation harness drives
`/query` once per golden question against a slow CPU model, so a timeout is
an expected operating condition, not a rare exceptional one.

`LLM_UNAVAILABLE_ERRORS` was built empirically, not guessed, against the two
LLM clients this app actually constructs; it now lives in `app.core.errors`
and is re-exported here under its original name. It moved because
`AgenticPipeline.answer()` gained a catch-all containment layer (so an
unexpected stage failure returns a friendly result instead of a bare 500),
and that layer must deliberately *re-raise* these specific errors so this
handler still sees them and still returns 503. See `app.core.errors` for the
full derivation and for why the tuple must stay narrow.
"""

import logging
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from opentelemetry import trace
from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.core.errors import LLM_UNAVAILABLE_ERRORS
from app.pipelines.types import Citation, PipelineResult, RetrievedChunk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])

__all__ = ["LLM_UNAVAILABLE_ERRORS", "QueryIn", "QueryOut", "get_pipeline", "query", "router"]


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
    # Imported lazily so simple-mode requests (and this module's unit
    # tests, which patch `get_pipeline` outright) never need CrewAI/LiteLLM
    # importable.
    from app.pipelines.agentic import AgenticPipeline
    return AgenticPipeline()


@router.post("/query", response_model=QueryOut)
def query(body: QueryIn) -> QueryOut:
    mode = body.mode or get_settings().pipeline_mode
    start = time.perf_counter()
    try:
        result: PipelineResult = get_pipeline(mode).answer(body.question, history=[])
    except LLM_UNAVAILABLE_ERRORS as exc:
        logger.error(
            "query failed: LLM backend unreachable or timed out mode=%s: %s",
            mode, exc,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "The upstream LLM service (Ollama) is unavailable or timed "
                "out. Please retry."
            ),
        ) from exc
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
