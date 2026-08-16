import logging
import threading

from fastapi import APIRouter, HTTPException

from app.api.schemas import JobOut
from app.db.repository import JobRepository
from app.db.session import get_session
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.post("", status_code=202)
def start_ingest() -> dict:
    with get_session() as s:
        job_id = JobRepository(s).create().id

    def _run() -> None:
        try:
            IngestionPipeline().run(job_id)
        except Exception:
            logger.exception("ingest job %s crashed", job_id)
            with get_session() as s:
                JobRepository(s).finish(job_id, "failed")

    threading.Thread(target=_run, name=f"ingest-{job_id}", daemon=True).start()
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=JobOut)
def job_status(job_id: str) -> JobOut:
    with get_session() as s:
        job = JobRepository(s).get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return JobOut.model_validate(job)
