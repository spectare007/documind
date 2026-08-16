"""The one tool the researcher agent gets: hybrid search over the PDF corpus.

The agent only ever sees the rendered text of the chunks, but the pipeline
needs the structured `RetrievedChunk`s (scores, titles, sections, pages) to
grade them and build citations. So the tool writes the raw chunks into a
caller-owned `buffer` list that `CrewStages.research` reads back after the
crew finishes -- the LLM's summary of what it found is discarded.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from app.pipelines.types import RetrievedChunk
from app.retrieval.retriever import HybridRetriever


class SearchInput(BaseModel):
    query: str = Field(description="standalone search query for the document knowledge base")


class DocumentSearchTool(BaseTool):
    """CrewAI tool wrapping `HybridRetriever`, with a side channel for results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "document_search"
    description: str = "Search the PDF knowledge base. Returns numbered text chunks."
    args_schema: type[BaseModel] = SearchInput

    # `SkipValidation` here keeps the retriever a dependency-injection seam,
    # exactly like `CrewStages(retriever=...)` and `SimplePipeline(retriever=...)`
    # which do no runtime type check at all: a plain `HybridRetriever`
    # annotation on a pydantic model becomes a hard isinstance() gate, so a
    # test double or an evaluation-harness stub would be rejected here even
    # though every other layer accepts it.
    retriever: SkipValidation[HybridRetriever]

    # `SkipValidation` is load-bearing, not decoration. `BaseTool` is a
    # pydantic v2 model, and pydantic validates a plain `list[RetrievedChunk]`
    # field by *rebuilding* the list: `DocumentSearchTool(buffer=buf).buffer
    # is buf` would be False, so every chunk the agent retrieved would be
    # appended to a copy nobody reads, and `research()` would silently fall
    # back to re-running retrieval itself. Skipping validation keeps the
    # caller's exact list object. Covered by
    # tests/unit/test_agentic_stages.py::test_tool_appends_to_the_callers_buffer.
    buffer: SkipValidation[list[RetrievedChunk]]

    # Per-request override for `retrieval_top_k`, threaded down from
    # `POST /api/v1/query`'s `top_k`. `None` keeps the configured default.
    top_k: int | None = None

    def _run(self, query: str) -> str:
        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        self.buffer.extend(chunks)
        if not chunks:
            return "No results found."
        return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
