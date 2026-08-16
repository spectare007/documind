"""Hybrid (dense + sparse) retriever over the PGVector chunk store."""

import logging

from llama_index.core import VectorStoreIndex

from app.core.config import get_settings
from app.pipelines.types import Citation, RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Wraps a LlamaIndex `VectorStoreIndex` to run hybrid PGVector queries
    and map results onto the app's own `RetrievedChunk` type.
    """

    def __init__(self, index: VectorStoreIndex | None = None) -> None:
        if index is None:
            from app.retrieval.vector_store import get_embed_model, get_vector_store
            index = VectorStoreIndex.from_vector_store(
                get_vector_store(), embed_model=get_embed_model()
            )
        self.index = index

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or get_settings().retrieval_top_k
        retriever = self.index.as_retriever(
            similarity_top_k=k,
            sparse_top_k=k,
            vector_store_query_mode="hybrid",
        )
        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []
        for node_with_score in retriever.retrieve(query):
            node = node_with_score.node
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            meta = node.metadata or {}
            chunks.append(RetrievedChunk(
                text=node.get_content(),
                score=float(node_with_score.score or 0.0),
                doc_id=str(meta.get("doc_id", "")),
                title=str(meta.get("title", "")),
                section_path=str(meta.get("section_path", "")),
                pages=list(meta.get("pages", []) or []),
            ))
        logger.info("retrieved %d chunks for query %r", len(chunks), query[:80])
        return chunks


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Collapse chunks into unique (title, section_path) citations, merging
    their page numbers. Order follows first appearance.
    """
    grouped: dict[tuple[str, str], Citation] = {}
    for c in chunks:
        key = (c.title, c.section_path)
        if key not in grouped:
            grouped[key] = Citation(title=c.title, section_path=c.section_path, pages=[])
        grouped[key].pages = sorted(set(grouped[key].pages) | set(c.pages))
    return list(grouped.values())
