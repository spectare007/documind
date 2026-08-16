import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import DocumentOut
from app.core.config import get_settings
from app.db.repository import DocumentRepository
from app.db.session import get_session
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    with get_session() as s:
        return [DocumentOut.model_validate(d) for d in DocumentRepository(s).list_all()]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str) -> DocumentOut:
    with get_session() as s:
        doc = DocumentRepository(s).get(doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        return DocumentOut.model_validate(doc)


@router.post("", response_model=DocumentOut, status_code=201)
def upload_document(file: UploadFile) -> DocumentOut:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.data_dir / Path(file.filename).name
    target.write_bytes(file.file.read())
    logger.info("uploaded %s", target.name)
    doc_id = IngestionPipeline().ingest_file(target)
    with get_session() as s:
        doc = DocumentRepository(s).get(doc_id)
        if doc is None:
            # Ingestion reported success but the ledger row isn't visible yet
            # (e.g. a mocked/async pipeline) -- respond with what we know.
            return DocumentOut(id=doc_id, filename=target.name, status="processing")
        return DocumentOut.model_validate(doc)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    with get_session() as s:
        if DocumentRepository(s).get(doc_id) is None:
            raise HTTPException(404, "document not found")
    IngestionPipeline().delete_document(doc_id)
