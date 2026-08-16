import logging
import threading
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import TableItem

from app.ingestion.types import ParsedDocument, RawChunk

logger = logging.getLogger(__name__)

_converter: DocumentConverter | None = None

# Guards both the lazy construction of the shared DocumentConverter and every convert()
# call made through it. DocumentConverter.convert() is not documented thread-safe, and
# ingestion jobs each run on their own thread, so PDF conversion is fully
# serialised here: a single shared converter used one document at a time is the right
# trade for this project, since correctness matters far more than ingestion throughput.
# Re-entrant so parse_pdf() can hold the lock across both the lazy-init check and the
# convert() call without deadlocking against _get_converter()'s own locking.
_converter_lock = threading.RLock()

# docling 2.120.x default tokenizer for HybridChunker when none is supplied.
# This counts tokens with all-MiniLM-L6-v2's tokenizer, not nomic-embed-text's
# (the actual embedding model, configured via Settings.embed_model). The 512
# cap is therefore enforced in a different tokenizer's token units. Safe in
# practice: nomic-embed-text's context window is 8192 tokens, far larger than
# any chunk this cap can produce, so the mismatch cannot silently truncate an
# embedding input.
_DEFAULT_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:  # fast path: avoid locking once warmed up
        with _converter_lock:
            if _converter is None:  # re-check inside the lock (double-checked locking)
                _converter = DocumentConverter()  # heavyweight: loads layout models on first use
    return _converter


def parse_pdf(path: Path, max_tokens: int = 512) -> ParsedDocument:
    with _converter_lock:  # serialise conversion; see _converter_lock comment above
        result = _get_converter().convert(str(path))
    doc = result.document
    title = (doc.name or path.stem).strip() or path.stem
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=_DEFAULT_TOKENIZER_MODEL, max_tokens=max_tokens
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    chunks: list[RawChunk] = []
    for chunk in chunker.chunk(doc):
        meta = chunk.meta
        headings = list(getattr(meta, "headings", None) or [])
        pages: set[int] = set()
        is_table = False
        for item in getattr(meta, "doc_items", None) or []:
            if isinstance(item, TableItem):
                is_table = True
            for prov in getattr(item, "prov", None) or []:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    pages.add(int(page_no))
        text = chunk.text.strip()
        if not text:
            continue
        chunks.append(RawChunk(
            text=text, section_path=headings, pages=sorted(pages), is_table=is_table
        ))

    page_count = len(getattr(doc, "pages", {}) or {})
    logger.info("parsed %s: %d pages, %d chunks", path.name, page_count, len(chunks))
    return ParsedDocument(title=title, page_count=page_count, chunks=chunks)
