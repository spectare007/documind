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
