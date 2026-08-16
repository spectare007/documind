"""Ingest all PDFs from data/documents into the vector store."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.logging import setup_logging          # noqa: E402
from app.db.repository import JobRepository         # noqa: E402
from app.db.session import get_session, init_db     # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402


def main() -> int:
    setup_logging()
    init_db()
    with get_session() as s:
        job_id = JobRepository(s).create().id
    IngestionPipeline().run(job_id)
    with get_session() as s:
        job = JobRepository(s).get(job_id)
        print(f"job {job_id}: {job.completed_documents} ok, {job.failed_documents} failed")
        return 1 if job.failed_documents else 0


if __name__ == "__main__":
    raise SystemExit(main())
