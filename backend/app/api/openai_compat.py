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

The pipeline run is submitted to a dedicated bounded executor
(`_get_stream_executor`) rather than run directly on this coroutine, so a
client that aborts or refreshes mid-stream (the normal case for a 30-60s
answer, not an edge case) doesn't leak the in-flight run forever: the
generator's `finally` clause (see `chat_completions`) attempts real
cancellation and, if the run is already executing and can't be interrupted,
attaches a done-callback so its eventual result is still retrieved and
logged instead of silently accumulating.

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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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


@lru_cache
def _get_stream_executor() -> ThreadPoolExecutor:
    """Bounded, dedicated thread pool for agentic pipeline runs started by
    the streaming chat endpoint.

    Deliberately separate from asyncio's shared default executor (which
    `asyncio.to_thread` -- used by this module's own non-streaming path and
    by `/api/v1/query` -- and every other unqualified `run_in_executor(None,
    ...)` call in the app also draws from). A `pipeline.answer()` already
    executing on a worker thread cannot be interrupted mid-LLM-call, so a
    client that disconnects mid-stream leaves its run occupying a thread
    until it finishes or `request_budget_seconds` (Task 10) gives up on it.
    Isolating that thread pool means repeated aborts can only ever exhaust
    *this* endpoint's capacity, never starve DB writes, ingestion, or other
    request handling that shares the default pool. Sized via
    `Settings.chat_stream_max_workers` so it's tunable without a code change.
    """
    return ThreadPoolExecutor(
        max_workers=get_settings().chat_stream_max_workers,
        thread_name_prefix="chat-stream",
    )


def _post_status(loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[str]", msg: str) -> None:
    """Thread-safe hand-off of a pipeline status update to the request's
    asyncio queue, called from the pipeline's worker thread.

    Guarded: a run whose stream has already been torn down (client
    disconnected, or in a test, the event loop it was bound to has since been
    closed) can still be executing on its own thread and calling
    `on_status`. `loop.call_soon_threadsafe` on a closed/cancelled loop
    raises `RuntimeError`; left unguarded, that would propagate straight
    into `AgenticPipeline`'s calling thread as an unrelated crash instead of
    the pipeline's own exceptions. Dropping a late status update is
    harmless -- by the time it could happen, nothing is left to render it.
    """
    try:
        loop.call_soon_threadsafe(queue.put_nowait, msg)
    except RuntimeError:
        logger.debug(
            "dropped a late pipeline status update: event loop is no longer running"
        )


def _reap_abandoned_run(cid: str, future: "asyncio.Future[PipelineResult]") -> None:
    """Done-callback for a pipeline run whose stream was torn down before it
    finished (client disconnected mid-stream).

    The run itself cannot be interrupted once its LLM call is in flight (see
    `_get_stream_executor`), so it keeps going until it naturally finishes or
    `request_budget_seconds` gives up on it. This callback exists so that:
    (1) the eventual result/exception is *retrieved*, so asyncio doesn't log
    an unrelated "Future exception was never retrieved" warning for it, and
    (2) the abandoned run is accounted for in the logs instead of silently
    vanishing with nobody left to consume it.
    """
    if future.cancelled():
        logger.info(
            "chat completion %s: abandoned pipeline run was cancelled before it started",
            cid,
        )
        return
    exc = future.exception()
    if exc is not None:
        logger.info(
            "chat completion %s: abandoned pipeline run (client already disconnected) "
            "finished with an error: %s", cid, exc,
        )
    else:
        logger.info(
            "chat completion %s: abandoned pipeline run (client already disconnected) "
            "finished normally", cid,
        )


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
        _post_status(loop, queue, msg)

    async def generate():
        yield _chunk_frame(cid, created, {"role": "assistant", "content": "<think>\n"})
        # Submitted directly against the executor (rather than via
        # `loop.run_in_executor`) so `raw_future` -- the real
        # `concurrent.futures.Future` -- is still in hand in `finally`.
        # `raw_future.cancel()` gives an accurate signal (True only if the
        # job hadn't started running yet); `task.cancel()` on the
        # asyncio-wrapped future does not -- it reports success
        # unconditionally regardless of whether the underlying thread is
        # already executing, which would make the disconnect log message
        # below lie about whether the run actually stopped.
        raw_future = _get_stream_executor().submit(pipeline.answer, question, history, on_status)
        task = asyncio.wrap_future(raw_future, loop=loop)
        get_status: asyncio.Task | None = None
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
                get_status = None
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
        finally:
            # Runs on every exit path, including a client disconnect: Starlette
            # tears this generator down via GeneratorExit (aclose()) or a
            # CancelledError thrown at the `await asyncio.wait(...)` point
            # above -- neither is caught by the except clauses above (both
            # are BaseException, not Exception), so this is the only place
            # that reliably sees "the stream is gone but the pipeline run may
            # still be going." On the normal-completion and handled-exception
            # paths above, `task` is already done by construction (we only
            # ever reach `task.result()` after it's in asyncio.wait's `done`
            # set), so the block below is a no-op there.
            if get_status is not None and not get_status.done():
                get_status.cancel()
            if not task.done():
                # raw_future.cancel() is the real signal: True only if the
                # executor hadn't started the job on a worker thread yet (in
                # which case it now never will), False if it's already
                # executing pipeline.answer() and therefore cannot be
                # interrupted mid-LLM-call. Either way, `task` -- chained to
                # `raw_future` by asyncio.wrap_future -- will still reach
                # `done()` on its own: immediately if truly cancelled, or
                # later with the real result/exception if not, which is why
                # the done-callback is attached unconditionally rather than
                # only in the "didn't cancel" branch. This is what accounts
                # for the run instead of leaking it silently; the pipeline's
                # own request_budget_seconds wall-clock cap (Task 10) bounds
                # how much longer an in-flight run can occupy its thread with
                # nobody left to consume the result.
                really_cancelled = raw_future.cancel()
                logger.info(
                    "chat completion %s: client disconnected mid-stream; pipeline run %s",
                    cid,
                    "was cancelled before it started" if really_cancelled else
                    "is already in flight on its worker thread and cannot be "
                    "interrupted -- it will stop within request_budget_seconds "
                    f"(~{get_settings().request_budget_seconds:.0f}s) at most",
                )
                task.add_done_callback(lambda f: _reap_abandoned_run(cid, f))

        while not queue.empty():  # drain any statuses that landed after the pipeline returned
            yield _chunk_frame(cid, created, {"content": f"{queue.get_nowait()}\n"})
        yield _chunk_frame(cid, created, {"content": "</think>\n\n"})
        text = result.answer + format_sources(result)
        for i in range(0, len(text), 48):
            yield _chunk_frame(cid, created, {"content": text[i:i + 48]})
        yield _chunk_frame(cid, created, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
