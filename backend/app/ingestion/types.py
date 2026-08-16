from pydantic import BaseModel, Field


class RawChunk(BaseModel):
    text: str
    section_path: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    is_table: bool = False


class ParsedDocument(BaseModel):
    title: str
    page_count: int
    chunks: list[RawChunk]
