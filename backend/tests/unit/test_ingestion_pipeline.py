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
