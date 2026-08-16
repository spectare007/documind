"""OpenAI-compatible chat API consumed by OpenWebUI.

Exposes `GET /v1/models` and `POST /v1/chat/completions` so any
OpenAI-client-compatible frontend (OpenWebUI in particular) can drive the
agentic RAG pipeline as if it were talking to a normal chat model. This is a
thin adapter over `app.api.query.get_pipeline` / `PipelineResult` (Task 9) --
it does not re-implement any pipeline logic.

--- Streaming status updates ---

The agentic pipeline is slow (30-60s on CPU) and calls `on_status(msg)` at
each stage boundary (router, rewrite, retrieve, grade, synthesize, check --
see `app.pipelines.agentic`). OpenWebUI renders any `<think>...</think>`
span in assistant content as a collapsible "thinking" panel, so status
updates are streamed as individual chunks *inside* an open `<think>` block
while the pipeline runs in a background thread; once it finishes, the block
is closed and the final answer + source list stream in afterwards. This is
the entire reason the chat UI feels alive during a long-running answer --
without it, OpenWebUI would just show nothing for up to a minute.

--- LLM-unavailable handling ---

`app.api.query.query()` catches `LLM_UNAVAILABLE_ERRORS` and returns a
structured 503. The non-streaming path here does the same, for parity with
that endpoint and because a plain JSON response has not been sent yet when
the error is raised, so a real HTTP error status is still possible.

The streaming path is different: by the time an LLM-unavailable error could
occur, `StreamingResponse` has already sent a 200 status and the opening
`<think>` chunk -- HTTP does not allow changing the status code after that.
So instead of letting the exception escape (which would just drop the
connection and look like a hang to the user), the generator catches
`LLM_UNAVAILABLE_ERRORS` *and* any other unexpected exception, emits the
failure as visible text inside the still-open `<think>` block (or, if the
think block was already closed, as a plain content chunk), and then ends the
stream cleanly with a `finish_reason: "stop"` chunk and `data: [DONE]` --
exactly like a normal completion, just with an apologetic message instead of
an answer. This keeps every stream well-formed for any OpenAI-compatible
client, which is more important than distinguishing "we failed" from "here is
the answer" at the transport level once streaming has already started.
"""

import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.query import get_pipeline
from app.core.config import get_settings
from app.core.errors import LLM_UNAVAILABLE_ERRORS
from app.pipelines.types import PipelineResult

logger = logging.getLogger(__name__)
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
    """Split an OpenAI `messages` array into (latest question, prior turns).

    The last `user` message is the question being asked now; every earlier
    `user`/`assistant` message becomes history, with any `<think>...</think>`
    status block stripped out of assistant turns (OpenWebUI round-trips our
    own thinking panel back to us as prior assistant content, and the
    pipeline should only ever see the real answer, not stage-status noise).
    `system` messages are dropped entirely -- the pipeline has no concept of
    a system prompt today.
    """
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
        try:
            result = await asyncio.to_thread(pipeline.answer, question, history)
        except LLM_UNAVAILABLE_ERRORS as exc:
            logger.error("chat completion failed: LLM backend unreachable or timed out: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "The upstream LLM service (Ollama) is unavailable or timed "
                    "out. Please retry."
                ),
            ) from exc
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
        try:
            while True:
                get_status = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_status, task}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_status in done:
                    yield _chunk_frame(cid, created, {"content": f"{get_status.result()}\n"})
                    continue
                get_status.cancel()
                break
            result: PipelineResult = task.result()
        except LLM_UNAVAILABLE_ERRORS as exc:
            logger.error(
                "streaming chat completion failed: LLM backend unreachable or timed out: %s", exc
            )
            yield _chunk_frame(
                cid, created,
                {"content": "⚠️ The upstream LLM service (Ollama) is unavailable "
                            "or timed out. Please retry.\n"},
            )
            yield _chunk_frame(cid, created, {"content": "</think>\n\n"})
            yield _chunk_frame(cid, created, {}, finish="stop")
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:  # noqa: BLE001 -- last-resort containment for an SSE
            # stream that has already sent a 200: surface the failure as
            # visible text and end cleanly rather than dropping the
            # connection, which OpenWebUI would otherwise show as a silent
            # hang with no explanation.
            logger.exception("streaming chat completion failed unexpectedly: %s", exc)
            yield _chunk_frame(
                cid, created,
                {"content": "⚠️ Something went wrong while generating this "
                            "answer. Please retry.\n"},
            )
            yield _chunk_frame(cid, created, {"content": "</think>\n\n"})
            yield _chunk_frame(cid, created, {}, finish="stop")
            yield "data: [DONE]\n\n"
            return

        while not queue.empty():  # drain any statuses that landed after the pipeline returned
            yield _chunk_frame(cid, created, {"content": f"{queue.get_nowait()}\n"})
        yield _chunk_frame(cid, created, {"content": "</think>\n\n"})
        text = result.answer + format_sources(result)
        for i in range(0, len(text), 48):
            yield _chunk_frame(cid, created, {"content": text[i:i + 48]})
        yield _chunk_frame(cid, created, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
