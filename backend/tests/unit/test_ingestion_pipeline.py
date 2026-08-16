from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.types import ParsedDocument, RawChunk


@pytest.fixture
def session_factory():
    from app.db.models import Base
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _parsed():
    return ParsedDocument(title="Doc", page_count=2, chunks=[
        RawChunk(text="alpha", section_path=["S1"], pages=[1]),
        RawChunk(text="beta", section_path=["S2"], pages=[2], is_table=True),
    ])


def _pipeline(session_factory, tmp_path, parse=None):
    from app.ingestion.pipeline import IngestionPipeline
    store, embed = MagicMock(), MagicMock()
    embed.get_text_embedding_batch.return_value = [[0.1] * 768, [0.2] * 768]
    p = IngestionPipeline(
        parse=parse or MagicMock(return_value=_parsed()),
        store=store, embed=embed,
        session_factory=session_factory, data_dir=tmp_path,
    )
    return p, store


def test_ingest_file_indexes_contextualized_chunks(session_factory, tmp_path):
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-fake")
    pipeline, store = _pipeline(session_factory, tmp_path)
    doc_id = pipeline.ingest_file(pdf)

    nodes = store.add.call_args.args[0]
    assert len(nodes) == 2
    assert nodes[0].text.startswith("[Doc > S1]\n\n")
    assert nodes[0].embedding == [0.1] * 768
    assert nodes[0].metadata["doc_id"] == doc_id
    assert nodes[1].metadata["is_table"] is True

    from app.db.repository import DocumentRepository
    with session_factory() as s:
        rec = DocumentRepository(s).get(doc_id)
        assert (rec.status, rec.chunk_count, rec.page_count) == ("completed", 2, 2)


def test_ingest_file_records_embedding_model_on_chunk_metadata(session_factory, tmp_path):
    """A future embedding-model change must be detectable rather than silently
    mixing vector spaces, so every chunk's metadata records which model built
    its vector (see `Settings.embed_model` and `doc/architecture.md`)."""
    from app.core.config import get_settings
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-fake")
    pipeline, store = _pipeline(session_factory, tmp_path)
    pipeline.ingest_file(pdf)

    nodes = store.add.call_args.args[0]
    assert all(n.metadata["embedding_model"] == get_settings().embed_model for n in nodes)


def test_run_skips_unchanged_and_isolates_failures(session_factory, tmp_path):
    good, bad = tmp_path / "good.pdf", tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF-1"); bad.write_bytes(b"%PDF-2")

    def parse(path, max_tokens=512):
        if "bad" in str(path):
            raise ValueError("corrupt pdf")
        return _parsed()

    pipeline, _ = _pipeline(session_factory, tmp_path, parse=MagicMock(side_effect=parse))
    from app.db.repository import JobRepository
    with session_factory() as s:
        job_id = JobRepository(s).create().id; s.commit()

    pipeline.run(job_id)

    from app.db.repository import DocumentRepository
    with session_factory() as s:
        docs = {d.filename: d for d in DocumentRepository(s).list_all()}
        job = JobRepository(s).get(job_id)
    assert docs["good.pdf"].status == "completed"
    assert docs["bad.pdf"].status == "failed" and "corrupt" in docs["bad.pdf"].error
    assert (job.completed_documents, job.failed_documents, job.status) == (1, 1, "completed")

    # second run: same hashes -> both skipped, no new parse calls for good.pdf
    calls_before = pipeline.parse.call_count
    with session_factory() as s:
        job2 = JobRepository(s).create().id; s.commit()
    pipeline.run(job2)
    assert pipeline.parse.call_count == calls_before + 1  # only failed doc retried


# --- re-ingest identity (B3) ---
#
# Document identity is the *filename*, not the content hash. Keying on the
# hash meant a changed file missed the lookup, got a brand-new row and a
# brand-new doc_id, and the "drop stale chunks" delete then targeted that new
# id -- a guaranteed no-op that left the previous version's chunks in the
# index forever and listed the same filename twice in GET /api/v1/documents.


def test_reingesting_a_changed_file_reuses_one_row_and_replaces_its_chunks(
    session_factory, tmp_path
):
    from app.db.repository import DocumentRepository

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-version-one")
    pipeline, store = _pipeline(session_factory, tmp_path)
    first_doc_id = pipeline.ingest_file(pdf)

    with session_factory() as s:
        first_sha = DocumentRepository(s).get(first_doc_id).sha256

    pdf.write_bytes(b"%PDF-version-two-with-different-bytes")
    second_doc_id = pipeline.ingest_file(pdf)

    assert second_doc_id == first_doc_id, "a changed file must keep its doc_id"

    with session_factory() as s:
        rows = DocumentRepository(s).list_all()
        assert [r.filename for r in rows] == ["a.pdf"], "exactly one row per filename"
        assert rows[0].id == first_doc_id
        assert rows[0].sha256 != first_sha, "sha256 must track the new bytes"
        assert rows[0].status == "completed"

    # The delete that drops the previous version's chunks must target the same
    # id the new chunks are inserted under, or it is a no-op and stale chunks
    # survive alongside the current ones.
    deleted_ids = [c.kwargs["ref_doc_id"] for c in store.delete.call_args_list]
    assert deleted_ids == [first_doc_id, first_doc_id]
    inserted_ids = {n.metadata["doc_id"] for n in store.add.call_args.args[0]}
    assert inserted_ids == {first_doc_id}


def test_reingesting_an_unchanged_file_is_skipped(session_factory, tmp_path):
    """The idempotency guarantee: identical bytes are never re-parsed."""
    from app.db.repository import JobRepository

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-unchanged")
    pipeline, _ = _pipeline(session_factory, tmp_path)

    for _ in range(2):
        with session_factory() as s:
            job_id = JobRepository(s).create().id; s.commit()
        pipeline.run(job_id)

    assert pipeline.parse.call_count == 1, "unchanged bytes must not be re-parsed"


def test_a_changed_file_is_not_skipped(session_factory, tmp_path):
    """The other half of the skip contract: a new hash on a known filename
    means the file was edited on disk and has to be re-ingested.
    """
    from app.db.repository import JobRepository

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-one")
    pipeline, _ = _pipeline(session_factory, tmp_path)
    with session_factory() as s:
        job_id = JobRepository(s).create().id; s.commit()
    pipeline.run(job_id)

    pdf.write_bytes(b"%PDF-two")
    with session_factory() as s:
        job2 = JobRepository(s).create().id; s.commit()
    pipeline.run(job2)

    assert pipeline.parse.call_count == 2


def test_two_files_with_identical_bytes_get_their_own_rows(session_factory, tmp_path):
    """SHA-256 is a change signal, not an identity, so it cannot be unique:
    a duplicated file is still two documents with two sets of citations.
    """
    from app.db.repository import DocumentRepository

    (tmp_path / "a.pdf").write_bytes(b"%PDF-same")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-same")
    pipeline, _ = _pipeline(session_factory, tmp_path)
    id_a = pipeline.ingest_file(tmp_path / "a.pdf")
    id_b = pipeline.ingest_file(tmp_path / "b.pdf")

    assert id_a != id_b
    with session_factory() as s:
        assert sorted(r.filename for r in DocumentRepository(s).list_all()) == [
            "a.pdf", "b.pdf",
        ]
