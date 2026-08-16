from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

AUTH = {"Authorization": "Bearer documind-dev-key"}

# Repo-root prompts/ dir: in production the container's cwd is /app with
# prompts/ mounted alongside it (docker-compose.yml), so Settings.prompts_dir
# defaults to a bare relative "prompts". Under pytest the cwd is backend/, so
# the lifespan's PromptManager needs to be pointed at the real directory.
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.db.session as db_session
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    db_session.get_engine.cache_clear()
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMIND_PROMPTS_DIR", str(PROMPTS_DIR))
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


def test_health_is_public(client):
    with patch("app.api.health._check_url", return_value=True):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_requires_auth(client):
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_ingest_creates_job_and_reports_status(client):
    with patch("app.api.ingest.IngestionPipeline") as pipe_cls:
        pipe_cls.return_value.run = MagicMock()
        r = client.post("/api/v1/ingest", headers=AUTH)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    r2 = client.get(f"/api/v1/ingest/{job_id}", headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["id"] == job_id


def test_upload_saves_pdf_and_lists_document(client, tmp_path):
    with patch("app.api.documents.IngestionPipeline") as pipe_cls:
        pipe_cls.return_value.ingest_file = MagicMock(return_value="doc123")
        r = client.post(
            "/api/v1/documents", headers=AUTH,
            files={"file": ("new.pdf", b"%PDF-1.7 fake", "application/pdf")},
        )
    assert r.status_code == 201
    assert (tmp_path / "new.pdf").exists()


def test_upload_rejects_non_pdf(client):
    r = client.post(
        "/api/v1/documents", headers=AUTH,
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 400
