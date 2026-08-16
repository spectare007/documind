import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def session():
    from app.db.models import Base
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        yield s


def test_document_lifecycle(session):
    from app.db.repository import DocumentRepository
    repo = DocumentRepository(session)
    doc = repo.create(filename="a.pdf", sha="abc123")
    assert doc.status == "pending"
    assert repo.get_by_sha("abc123").id == doc.id
    repo.mark_processing(doc.id)
    assert repo.get(doc.id).status == "processing"
    repo.mark_completed(doc.id, page_count=3, chunk_count=12)
    got = repo.get(doc.id)
    assert (got.status, got.page_count, got.chunk_count) == ("completed", 3, 12)
    assert got.ingested_at is not None


def test_lookup_by_filename_and_sha_update(session):
    """Filename is the identity; sha256 is a mutable change signal on it."""
    from app.db.repository import DocumentRepository
    repo = DocumentRepository(session)
    doc = repo.create(filename="a.pdf", sha="old")
    assert repo.get_by_filename("a.pdf").id == doc.id
    assert repo.get_by_filename("missing.pdf") is None
    repo.update_sha(doc.id, "new")
    assert repo.get(doc.id).sha256 == "new"
    assert repo.get_by_sha("old") is None


def test_update_sha_raises_for_unknown_id(session):
    from app.db.repository import DocumentRepository, RecordNotFoundError
    repo = DocumentRepository(session)
    with pytest.raises(RecordNotFoundError):
        repo.update_sha("does-not-exist", "abc")


def test_mark_failed_records_error(session):
    from app.db.repository import DocumentRepository
    repo = DocumentRepository(session)
    doc = repo.create(filename="bad.pdf", sha="ffff")
    repo.mark_failed(doc.id, error="parse error")
    got = repo.get(doc.id)
    assert got.status == "failed" and got.error == "parse error"


def test_job_progress(session):
    from app.db.repository import JobRepository
    repo = JobRepository(session)
    job = repo.create()
    assert job.status == "running"
    repo.update_progress(job.id, completed=2, failed=1)
    repo.finish(job.id, status="completed")
    got = repo.get(job.id)
    assert (got.completed_documents, got.failed_documents, got.status) == (2, 1, "completed")
    assert got.finished_at is not None


def test_document_mutator_raises_for_unknown_id(session):
    from app.db.repository import DocumentRepository, RecordNotFoundError
    repo = DocumentRepository(session)
    with pytest.raises(RecordNotFoundError):
        repo.mark_processing("does-not-exist")


def test_job_mutator_raises_for_unknown_id(session):
    from app.db.repository import JobRepository, RecordNotFoundError
    repo = JobRepository(session)
    with pytest.raises(RecordNotFoundError):
        repo.update_progress("does-not-exist", completed=1, failed=0)
