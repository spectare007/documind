import json
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
