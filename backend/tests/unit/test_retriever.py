from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode


def _node(nid, text, score, **meta):
    base = {"doc_id": "d1", "title": "Doc", "section_path": "S", "pages": [1], "is_table": False}
    base.update(meta)
    return NodeWithScore(node=TextNode(id_=nid, text=text, metadata=base), score=score)


def test_retrieve_maps_nodes_and_dedupes():
    from app.retrieval.retriever import HybridRetriever
    index = MagicMock()
    index.as_retriever.return_value.retrieve.return_value = [
        _node("a", "alpha", 0.9), _node("a", "alpha", 0.8), _node("b", "beta", 0.7, title="Doc2"),
    ]
    chunks = HybridRetriever(index=index).retrieve("q", top_k=5)
    assert [c.text for c in chunks] == ["alpha", "beta"]
    assert chunks[0].score == 0.9 and chunks[1].title == "Doc2"
    # Regression guard (review Minor finding): pin the actual query-mode
    # kwarg passed to `as_retriever`, not just its output -- a silent
    # regression to dense-only retrieval would otherwise still pass this
    # test, even though hybrid search is a headline feature.
    _, kwargs = index.as_retriever.call_args
    assert kwargs["vector_store_query_mode"] == "hybrid"
    assert kwargs["similarity_top_k"] == 5
    assert kwargs["sparse_top_k"] == 5


def test_build_citations_unique():
    from app.pipelines.types import RetrievedChunk
    from app.retrieval.retriever import build_citations
    chunks = [
        RetrievedChunk(text="x", score=1.0, doc_id="d", title="Doc", section_path="A", pages=[1]),
        RetrievedChunk(text="y", score=0.9, doc_id="d", title="Doc", section_path="A", pages=[2]),
        RetrievedChunk(text="z", score=0.8, doc_id="d", title="Doc", section_path="B", pages=[3]),
    ]
    cites = build_citations(chunks)
    assert [(c.title, c.section_path) for c in cites] == [("Doc", "A"), ("Doc", "B")]
    assert cites[0].pages == [1, 2]
