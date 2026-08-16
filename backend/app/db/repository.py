from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentRecord, IngestJob, utcnow


class RecordNotFoundError(LookupError):
    """Raised when a repository mutator is called with an id that has no matching row."""


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, filename: str, sha: str) -> DocumentRecord:
        doc = DocumentRecord(filename=filename, sha256=sha)
        self.session.add(doc)
        self.session.flush()
        return doc

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self.session.get(DocumentRecord, doc_id)

    def get_by_sha(self, sha: str) -> DocumentRecord | None:
        """Find a row by content hash.

        Not the identity lookup -- `get_by_filename` is (see `DocumentRecord`)
        -- and not unique, since two files may hold identical bytes. Kept as a
        content-addressed query for diagnostics and duplicate detection.
        """
        return self.session.scalar(select(DocumentRecord).where(DocumentRecord.sha256 == sha))

    def get_by_filename(self, filename: str) -> DocumentRecord | None:
        """Look a document up by its identity (see `DocumentRecord`)."""
        return self.session.scalar(
            select(DocumentRecord).where(DocumentRecord.filename == filename)
        )

    def list_all(self) -> list[DocumentRecord]:
        return list(self.session.scalars(select(DocumentRecord).order_by(DocumentRecord.filename)))

    def _require(self, doc_id: str) -> DocumentRecord:
        doc = self.get(doc_id)
        if doc is None:
            raise RecordNotFoundError(f"DocumentRecord not found: {doc_id}")
        return doc

    def update_sha(self, doc_id: str, sha: str) -> None:
        """Record the bytes this row is currently built from.

        Called when a file changed on disk so its existing row is re-pointed
        at the new content instead of a second row being created for it.
        """
        self._require(doc_id).sha256 = sha
        self.session.flush()

    def mark_processing(self, doc_id: str) -> None:
        self._require(doc_id).status = "processing"
        self.session.flush()

    def mark_completed(self, doc_id: str, page_count: int, chunk_count: int) -> None:
        doc = self._require(doc_id)
        doc.status, doc.page_count, doc.chunk_count = "completed", page_count, chunk_count
        doc.error, doc.ingested_at = None, utcnow()
        self.session.flush()

    def mark_failed(self, doc_id: str, error: str) -> None:
        doc = self._require(doc_id)
        doc.status, doc.error = "failed", error[:2000]
        self.session.flush()

    def delete(self, doc_id: str) -> None:
        doc = self.get(doc_id)
        if doc is not None:
            self.session.delete(doc)
            self.session.flush()


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self) -> IngestJob:
        job = IngestJob()
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: str) -> IngestJob | None:
        return self.session.get(IngestJob, job_id)

    def _require(self, job_id: str) -> IngestJob:
        job = self.get(job_id)
        if job is None:
            raise RecordNotFoundError(f"IngestJob not found: {job_id}")
        return job

    def set_total(self, job_id: str, total: int) -> None:
        self._require(job_id).total_documents = total
        self.session.flush()

    def update_progress(self, job_id: str, completed: int, failed: int) -> None:
        job = self._require(job_id)
        job.completed_documents, job.failed_documents = completed, failed
        self.session.flush()

    def finish(self, job_id: str, status: str, error: str | None = None) -> None:
        job = self._require(job_id)
        job.status, job.finished_at = status, utcnow()
        if error is not None:
            job.error = error
        self.session.flush()

    def reconcile_interrupted(self) -> list[str]:
        """Fail every job still `running`, on the assumption it was orphaned
        by a process restart.

        Ingest jobs run on a daemon thread (`app.api.ingest.start_ingest`), so
        a restart kills the thread mid-run without ever reaching `finish()`,
        leaving its row `running` forever with no worker left to finish it.
        Called once at startup (`app.main.lifespan`) so a stuck row surfaces
        as an honest, explained `failed` status instead of looking like it is
        still in progress indefinitely. Returns the ids of the jobs it failed.
        """
        stuck = list(
            self.session.scalars(select(IngestJob).where(IngestJob.status == "running"))
        )
        for job in stuck:
            job.status = "failed"
            job.error = "Interrupted by a server restart before the job could finish."
            job.finished_at = utcnow()
        self.session.flush()
        return [job.id for job in stuck]
