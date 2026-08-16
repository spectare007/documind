import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.repository import DocumentRepository, JobRepository
from app.ingestion.contextualizer import contextualize
from app.ingestion.types import ParsedDocument

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        parse: Callable[..., ParsedDocument] | None = None,
        store=None,
        embed=None,
        session_factory: sessionmaker | None = None,
        data_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        if parse is None:
            from app.ingestion.preprocessor import parse_pdf
            parse = parse_pdf
        if store is None:
            from app.retrieval.vector_store import get_vector_store
            store = get_vector_store()
        if embed is None:
            from app.retrieval.vector_store import get_embed_model
            embed = get_embed_model()
        if session_factory is None:
            from app.db.session import get_engine
            session_factory = sessionmaker(bind=get_engine())
        self.parse, self.store, self.embed = parse, store, embed
        self.session_factory = session_factory
        self.data_dir = data_dir or settings.data_dir
        self.chunk_max_tokens = settings.chunk_max_tokens

    def run(self, job_id: str) -> None:
        pdfs = sorted(self.data_dir.glob("*.pdf"))
        completed = failed = 0
        with self.session_factory() as s:
            JobRepository(s).set_total(job_id, len(pdfs))
            s.commit()
        for path in pdfs:
            try:
                if self._already_ingested(path):
                    logger.info("skipping unchanged %s", path.name)
                    completed += 1
                else:
                    self.ingest_file(path)
                    completed += 1
            except Exception as exc:  # per-document isolation
                logger.exception("ingestion failed for %s", path.name)
                failed += 1
                self._mark_failed(path, str(exc))
            with self.session_factory() as s:
                JobRepository(s).update_progress(job_id, completed, failed)
                s.commit()
        with self.session_factory() as s:
            JobRepository(s).finish(job_id, "completed")
            s.commit()

    def _already_ingested(self, path: Path) -> bool:
        """True only if this filename is already `completed` at these exact bytes.

        The hash is the change signal, not the identity: a row that exists for
        this filename but records a different hash means the file was edited on
        disk and has to be re-ingested into that same row.
        """
        sha = _sha256(path)
        with self.session_factory() as s:
            existing = DocumentRepository(s).get_by_filename(path.name)
            return (
                existing is not None
                and existing.status == "completed"
                and existing.sha256 == sha
            )

    def _ledger_row(self, path: Path, sha: str) -> str:
        """Return the stable `doc_id` for `path`, creating its row if needed.

        Document identity is the filename, so there is exactly one ledger row
        per file for the life of the corpus and its id survives every
        re-ingest. A changed file updates `sha256` on that row in place rather
        than getting a second row, which is what lets `ingest_file` delete the
        previous version's chunks (they are indexed under this same id).
        """
        with self.session_factory() as s:
            repo = DocumentRepository(s)
            doc = repo.get_by_filename(path.name)
            if doc is None:
                doc = repo.create(filename=path.name, sha=sha)
            doc_id = doc.id
            repo.update_sha(doc_id, sha)
            s.commit()
        return doc_id

    def _mark_failed(self, path: Path, error: str) -> None:
        doc_id = self._ledger_row(path, _sha256(path))
        with self.session_factory() as s:
            DocumentRepository(s).mark_failed(doc_id, error)
            s.commit()

    def ingest_file(self, path: Path) -> str:
        doc_id = self._ledger_row(path, _sha256(path))
        with self.session_factory() as s:
            DocumentRepository(s).mark_processing(doc_id)
            s.commit()

        parsed = self.parse(path, max_tokens=self.chunk_max_tokens)
        texts = [contextualize(c, parsed.title) for c in parsed.chunks]
        embeddings = self.embed.get_text_embedding_batch(texts, show_progress=False)

        nodes = []
        for chunk, text, emb in zip(parsed.chunks, texts, embeddings):
            nodes.append(
                TextNode(
                    text=text,
                    embedding=emb,
                    metadata={
                        "doc_id": doc_id,
                        "title": parsed.title,
                        "filename": path.name,
                        "section_path": " > ".join(chunk.section_path),
                        "pages": chunk.pages,
                        "is_table": chunk.is_table,
                    },
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                    },
                )
            )
        # `doc_id` is stable across re-ingests (one ledger row per filename),
        # so this really does remove the previous version's chunks before the
        # new ones go in. It is a no-op only on a genuinely first ingest.
        self.store.delete(ref_doc_id=doc_id)
        self.store.add(nodes)

        with self.session_factory() as s:
            DocumentRepository(s).mark_completed(doc_id, parsed.page_count, len(nodes))
            s.commit()
        logger.info("ingested %s: %d chunks", path.name, len(nodes))
        return doc_id

    def delete_document(self, doc_id: str) -> None:
        self.store.delete(ref_doc_id=doc_id)
        with self.session_factory() as s:
            DocumentRepository(s).delete(doc_id)
            s.commit()
