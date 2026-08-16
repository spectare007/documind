import logging
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from app.ingestion.types import ParsedDocument, RawChunk

logger = logging.getLogger(__name__)

_converter: DocumentConverter | None = None

# docling 2.120.x default tokenizer for HybridChunker when none is supplied.
_DEFAULT_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:  # heavyweight: loads layout models on first use
        _converter = DocumentConverter()
    return _converter


def parse_pdf(path: Path, max_tokens: int = 512) -> ParsedDocument:
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
            if type(item).__name__ == "TableItem":
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
