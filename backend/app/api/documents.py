import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.api.schemas import DocumentOut
from app.core.config import get_settings
from app.db.repository import DocumentRepository
from app.db.session import get_session
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# The PDF magic header ("%PDF", ASCII 0x25 0x50 0x44 0x46), checked against
# the actual uploaded bytes rather than trusting the `.pdf` filename suffix
# alone (fix for a review finding: a `.pdf`-named upload with arbitrary
# content was written straight into `data/documents/` and handed to the
# ingestion pipeline unchecked).
_PDF_MAGIC = b"%PDF"
# Read/write in fixed-size chunks so peak memory for a single upload stays
# bounded regardless of file size, instead of `file.file.read()`'s previous
# unbounded single read (fix for a review finding).
_UPLOAD_CHUNK_BYTES = 1024 * 1024


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
    """Save one uploaded PDF into `data_dir` and ingest it.

    Three checks run before any bytes are handed to the ingestion pipeline
    (all three are fixes for review findings):

    1. Filename must end in `.pdf` (unchanged, cheap first-pass filter).
    2. The bytes actually read back must start with the PDF magic header --
       a `.pdf`-named upload with unrelated content is rejected with 400
       rather than silently ingested.
    3. Total size is capped at `Settings.max_upload_bytes`, enforced while
       streaming (not after buffering the whole file), so an oversized
       upload is rejected with 413 without ever holding more than one chunk
       of it in memory. A partially written file from a rejected upload is
       always cleaned up.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "only PDF files are accepted")

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.data_dir / Path(file.filename).name

    total = 0
    header_checked = False
    try:
        with target.open("wb") as out:
            while chunk := file.file.read(_UPLOAD_CHUNK_BYTES):
                if not header_checked:
                    if not chunk.startswith(_PDF_MAGIC):
                        raise HTTPException(
                            status.HTTP_400_BAD_REQUEST,
                            "file content is not a valid PDF (missing %PDF header)",
                        )
                    header_checked = True
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"upload exceeds the {settings.max_upload_bytes}-byte limit",
                    )
                out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception:
        target.unlink(missing_ok=True)
        raise

    if total == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "uploaded file is empty")

    logger.info("uploaded %s (%d bytes)", target.name, total)
    doc_id = IngestionPipeline().ingest_file(target)
    with get_session() as s:
        return DocumentOut.model_validate(DocumentRepository(s).get(doc_id))


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> None:
    with get_session() as s:
        if DocumentRepository(s).get(doc_id) is None:
            raise HTTPException(404, "document not found")
    IngestionPipeline().delete_document(doc_id)
