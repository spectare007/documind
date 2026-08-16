import asyncio
import json
import logging
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.pipelines.types import Citation, PipelineResult

AUTH = {"Authorization": "Bearer documind-dev-key"}

# Repo-root prompts/ dir: in production the container's cwd is /app with
# prompts/ mounted alongside it (docker-compose.yml), so Settings.prompts_dir
# defaults to a bare relative "prompts". Under pytest the cwd is backend/, so
# the lifespan's PromptManager needs to be pointed at the real directory --
# same convention as tests/unit/test_api_query.py and test_api_documents.py.
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.db.session as db_session
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    db_session.get_engine.cache_clear()
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMIND_PROMPTS_DIR", str(PROMPTS_DIR))
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


def _result():
    return PipelineResult(answer="Total is 100.", grounded=True,
                          citations=[Citation(title="Invoice", section_path="Summary", pages=[1])])


def test_models_endpoint(client):
    r = client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "agentic-rag"


def test_extract_question_and_history():
    from app.api.openai_compat import extract_question_and_history
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "<think>hmm</think>first a"},
        {"role": "user", "content": "second q"},
    ]
    q, hist = extract_question_and_history(messages)
    assert q == "second q"
    assert hist == [{"role": "user", "content": "first q"},
                    {"role": "assistant", "content": "first a"}]


def test_chat_completion_non_streaming(client):
    with patch("app.api.openai_compat.get_pipeline") as gp:
        gp.return_value.answer = MagicMock(return_value=_result())
        r = client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "agentic-rag", "stream": False,
            "messages": [{"role": "user", "content": "total?"}],
        })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert "Total is 100." in content and "**Sources:**" in content
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming(client):
    with patch("app.api.openai_compat.get_pipeline") as gp:
        def fake_answer(question, history, on_status=None):
            if on_status:
                on_status("Routing query…")
            return _result()
        gp.return_value.answer = fake_answer
        r = client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "agentic-rag", "stream": True,
            "messages": [{"role": "user", "content": "total?"}],
        })
    assert r.status_code == 200
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(l[6:]) for l in lines[:-1]]
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    full = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert "<think>" in full and "Routing query…" in full and "</think>" in full
    assert "Total is 100." in full
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_empty_messages_rejected(client):
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "agentic-rag", "messages": []})
    assert r.status_code == 422


def test_format_sources_grounded_false_warns():
    result = PipelineResult(
        answer="Total is 100.", grounded=False,
        citations=[Citation(title="Invoice", section_path="Summary", pages=[1])],
    )
    from app.api.openai_compat import format_sources
    out = format_sources(result)
    assert "**Sources:**" in out
    assert "groundedness check did not pass" in out


def test_format_sources_grounded_none_has_no_warning():
    result = PipelineResult(
        answer="Total is 100.", grounded=None,
        citations=[Citation(title="Invoice", section_path="Summary", pages=[1])],
    )
    from app.api.openai_compat import format_sources
    out = format_sources(result)
    assert "**Sources:**" in out
    assert "groundedness check did not pass" not in out


def test_stream_disconnect_accounts_for_abandoned_pipeline_run(caplog):
    """Regression test for a review finding: `pipeline.answer()` runs in a
    background thread while the SSE generator streams status updates; if the
    client disconnects mid-stream, Starlette tears the generator down (a
    CancelledError thrown at the `await asyncio.wait(...)` suspend point, or
    a GeneratorExit via `aclose()`). The pending executor future used to be
    left completely unattended -- no cancellation attempted, no accounting,
    and (for the un-cancellable in-flight case) no way to ever learn the run
    finished. `TestClient` doesn't model a real client disconnect, so this
    drives the async generator directly and interrupts it with
    `asyncio.wait_for`'s timeout-driven cancellation, which throws
    `CancelledError` at exactly that suspend point -- the same mechanism
    Starlette itself uses.
    """
    from app.api.openai_compat import ChatCompletionRequest, chat_completions

    started = threading.Event()
    finish = threading.Event()

    def fake_answer(question, history, on_status=None):
        started.set()
        if on_status:
            on_status("Routing query…")
        finish.wait(timeout=2)  # simulate an LLM call still in flight
        return _result()

    async def run():
        with patch("app.api.openai_compat.get_pipeline") as gp:
            gp.return_value.answer = fake_answer
            body = ChatCompletionRequest(
                model="agentic-rag", stream=True,
                messages=[{"role": "user", "content": "total?"}],
            )
            response = await chat_completions(body)
            agen = response.body_iterator
            opening = await agen.__anext__()
            assert "<think>" in opening
            status_chunk = await agen.__anext__()
            assert "Routing query" in status_chunk
            assert started.wait(timeout=1), "pipeline run never started"

            with caplog.at_level(logging.INFO, logger="app.api.openai_compat"):
                with pytest.raises(asyncio.TimeoutError):
                    # The pipeline is still blocked in finish.wait(); this
                    # times out while the generator is suspended inside
                    # `await asyncio.wait(...)` in chat_completions.generate,
                    # and wait_for's cancellation throws the same
                    # CancelledError a real client disconnect would.
                    await asyncio.wait_for(agen.__anext__(), timeout=0.1)
                # Belt-and-braces: a real disconnect also calls aclose();
                # confirm the (now-exhausted) generator tolerates that too,
                # i.e. no exception escapes either teardown path.
                await agen.aclose()

                finish.set()  # let the abandoned run finish so its callback fires
                # The done-callback runs on *this* loop (asyncio.wrap_future
                # chains back to it); it must still be open when the
                # executor thread's real completion arrives, so poll from
                # inside this coroutine instead of a fixed sleep after
                # asyncio.run() would have already closed the loop.
                for _ in range(100):
                    if any("abandoned pipeline run" in r.getMessage()
                           for r in caplog.records):
                        break
                    await asyncio.sleep(0.02)

    asyncio.run(run())

    messages = [r.getMessage() for r in caplog.records]
    assert any("disconnected mid-stream" in m for m in messages), messages
    assert any(
        "already in flight on its worker thread and cannot be interrupted" in m
        for m in messages
    ), messages
    assert any("abandoned pipeline run" in m for m in messages), messages
    assert any("finished normally" in m for m in messages), messages


def test_post_status_swallows_a_closed_event_loop_instead_of_raising():
    """The on_status callback runs on the pipeline's worker thread and must
    never raise into it -- a status update arriving after the event loop it
    was bound to has been closed (e.g. this run outlived its request) would
    otherwise surface as an unrelated crash inside AgenticPipeline.
    """
    from app.api.openai_compat import _post_status

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    loop.close()
    assert loop.is_closed()

    _post_status(loop, queue, "late status")  # must not raise


def test_reap_abandoned_run_logs_each_outcome_without_raising(caplog):
    from app.api.openai_compat import _reap_abandoned_run

    async def _outcomes():
        cancelled = asyncio.get_running_loop().create_future()
        cancelled.cancel()

        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(RuntimeError("boom"))

        succeeded = asyncio.get_running_loop().create_future()
        succeeded.set_result(_result())

        with caplog.at_level(logging.INFO, logger="app.api.openai_compat"):
            _reap_abandoned_run("cid-1", cancelled)
            _reap_abandoned_run("cid-2", failed)
            _reap_abandoned_run("cid-3", succeeded)

    asyncio.run(_outcomes())

    messages = [r.getMessage() for r in caplog.records]
    assert any("cancelled before it started" in m for m in messages), messages
    assert any("finished with an error" in m for m in messages), messages
    assert any("finished normally" in m for m in messages), messages
