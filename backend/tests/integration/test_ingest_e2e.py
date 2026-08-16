import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="integration disabled"),
]


def test_full_ingest_and_ledger():
    from app.db.repository import JobRepository
    from app.db.session import get_session, init_db
    from app.ingestion.pipeline import IngestionPipeline

    init_db()
    with get_session() as s:
        job_id = JobRepository(s).create().id
    pipeline = IngestionPipeline()
    if not sorted(pipeline.data_dir.glob("*.pdf")):
        pytest.skip("no local PDFs")
    pipeline.run(job_id)
    with get_session() as s:
        job = JobRepository(s).get(job_id)
        assert job.status == "completed"
        assert job.completed_documents >= 1
