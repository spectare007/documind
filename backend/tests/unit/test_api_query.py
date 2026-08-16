from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.pipelines.types import Citation, PipelineResult

AUTH = {"Authorization": "Bearer documind-dev-key"}

# Repo-root prompts/ dir: in production the container's cwd is /app with
# prompts/ mounted alongside it (docker-compose.yml), so Settings.prompts_dir
# defaults to a bare relative "prompts". Under pytest the cwd is backend/, so
# the lifespan's PromptManager needs to be pointed at the real directory --
# same convention as tests/unit/test_api_documents.py.
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


def test_query_simple_mode(client):
    result = PipelineResult(answer="42 [Doc, A]", citations=[Citation(title="Doc")])
    with patch("app.api.query.get_pipeline") as gp:
        gp.return_value.answer = MagicMock(return_value=result)
        r = client.post("/api/v1/query", headers=AUTH,
                        json={"question": "meaning of life?", "mode": "simple"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "42 [Doc, A]"
    assert body["mode"] == "simple"
    assert "latency_ms" in body and "trace_id" in body


def test_query_validates_empty_question(client):
    r = client.post("/api/v1/query", headers=AUTH, json={"question": "  "})
    assert r.status_code == 422


def test_query_llm_connection_timeout_yields_503_not_500(client):
    """Regression test for a review finding: an unwrapped LLM connectivity
    failure used to propagate as FastAPI's generic unhandled 500 (no
    structured body, no route-level log). `SimplePipeline`'s LLM
    (`llama_index.llms.ollama.Ollama`) calls Ollama via `httpx` directly and
    lets connect/read timeouts and connection failures surface as raw
    `httpx.TransportError` subclasses -- confirmed empirically against the
    installed client, not assumed. See `app/api/query.py`'s module
    docstring for the full verification.
    """
    with patch("app.api.query.get_pipeline") as gp:
        gp.return_value.answer = MagicMock(
            side_effect=httpx.ConnectTimeout("timed out")
        )
        r = client.post("/api/v1/query", headers=AUTH,
                        json={"question": "meaning of life?", "mode": "simple"})
    assert r.status_code == 503
    body = r.json()
    assert "ollama" in body["detail"].lower()


def test_query_llm_connection_error_yields_503_not_500(client):
    """Same regression, for the other LLM client this app constructs:
    `AgenticPipeline`'s `crewai.LLM` re-raises both connection-refused and
    timeout failures as a plain builtin `ConnectionError` (confirmed
    empirically -- CrewAI catches the underlying `openai.APIConnectionError`/
    `APITimeoutError` and wraps it). Must not be swallowed as a bare
    `except Exception` -- it's specifically caught alongside the httpx
    family in `app.api.query.LLM_UNAVAILABLE_ERRORS`.
    """
    with patch("app.api.query.get_pipeline") as gp:
        gp.return_value.answer = MagicMock(
            side_effect=ConnectionError("Failed to connect to OpenAI API: Connection error.")
        )
        r = client.post("/api/v1/query", headers=AUTH,
                        json={"question": "meaning of life?", "mode": "agentic"})
    assert r.status_code == 503
    body = r.json()
    assert "ollama" in body["detail"].lower()


def test_query_programming_error_still_yields_500():
    """A bare `except Exception` around the pipeline call would also swallow
    real bugs (e.g. a `KeyError` in prompt formatting) and misreport them as
    "upstream unavailable". `LLM_UNAVAILABLE_ERRORS` must not catch this.
    """
    from app.api import query as query_module
    assert not issubclass(KeyError, query_module.LLM_UNAVAILABLE_ERRORS)
    assert not issubclass(ValueError, query_module.LLM_UNAVAILABLE_ERRORS)
