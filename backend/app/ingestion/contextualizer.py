from app.ingestion.types import RawChunk


def contextualize(chunk: RawChunk, doc_title: str) -> str:
    parts = [doc_title, *chunk.section_path]
    header = " > ".join(p.strip() for p in parts if p and p.strip())
    if chunk.is_table:
        header += " | table"
    return f"[{header}]\n\n{chunk.text}"
