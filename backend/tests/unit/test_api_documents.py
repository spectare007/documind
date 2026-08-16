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


def test_lifespan_syncs_then_refreshes_prompts(monkeypatch, tmp_path):
    """Finding 1 regression test: the lifespan must call sync_to_phoenix()
    and then refresh_from_phoenix() on every app startup, so a prompt edited
    in the Phoenix UI takes effect for this process without a restart.
    """
    import app.db.session as db_session
    engine = create_engine(f"sqlite:///{tmp_path}/test_lifespan.db")
    db_session.get_engine.cache_clear()
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(tmp_path))

    import app.observability.prompts as prompts_module
    fake_manager = MagicMock()
    call_order = []
    fake_manager.sync_to_phoenix.side_effect = lambda: call_order.append("sync")
    fake_manager.refresh_from_phoenix.side_effect = lambda: call_order.append("refresh")

    with patch.object(prompts_module, "get_prompt_manager", return_value=fake_manager):
        from app.main import create_app
        with TestClient(create_app()):
            pass

    fake_manager.sync_to_phoenix.assert_called_once()
    fake_manager.refresh_from_phoenix.assert_called_once()
    assert call_order == ["sync", "refresh"]


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
    # The real IngestionPipeline.ingest_file creates+commits the ledger row
    # before returning the id, and the endpoint reads it back in a fresh
    # session. Mimic that invariant here instead of just returning a bare
    # id string, so the endpoint's strict re-fetch-and-validate path is
    # exercised the way it will actually run in production.
    from app.db.repository import DocumentRepository
    from app.db.session import get_session

    def _fake_ingest_file(path):
        with get_session() as s:
            doc = DocumentRepository(s).create(filename=path.name, sha="deadbeef")
            return doc.id

    with patch("app.api.documents.IngestionPipeline") as pipe_cls:
        pipe_cls.return_value.ingest_file = MagicMock(side_effect=_fake_ingest_file)
        r = client.post(
            "/api/v1/documents", headers=AUTH,
            files={"file": ("new.pdf", b"%PDF-1.7 fake", "application/pdf")},
        )
    assert r.status_code == 201
    assert (tmp_path / "new.pdf").exists()
    body = r.json()
    assert body["filename"] == "new.pdf"
    assert body["status"] == "pending"
    assert body["id"]

    r2 = client.get("/api/v1/documents", headers=AUTH)
    assert r2.status_code == 200
    assert any(d["id"] == body["id"] and d["filename"] == "new.pdf" for d in r2.json())


def test_upload_rejects_non_pdf(client):
    r = client.post(
        "/api/v1/documents", headers=AUTH,
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 400
