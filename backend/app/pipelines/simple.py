"""Naive (non-agentic) RAG pipeline: retrieve then synthesize.

This is also the RAGAs evaluation baseline that the agentic pipeline
(`app.pipelines.agentic`) is compared against, so its behavior must stay
simple and deterministic: a single retrieval pass, a single generation
pass, no grading, no rewriting, no self-correction loop.
"""

import logging

from app.observability.prompts import get_prompt_manager
from app.pipelines.types import PipelineResult, RetrievedChunk, StatusCallback
from app.retrieval.retriever import HybridRetriever, build_citations

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the document knowledge base for that question."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(c.text for c in chunks)


class SimplePipeline:
    """Naive RAG: retrieve -> synthesize. Also the RAGAs baseline."""

    def __init__(self, retriever: HybridRetriever | None = None, llm=None) -> None:
        self.retriever = retriever or HybridRetriever()
        if llm is None:
            from app.core.config import get_settings
            from llama_index.llms.ollama import Ollama
            s = get_settings()
            llm = Ollama(
                model=s.llm_model,
                base_url=s.ollama_base_url,
                request_timeout=s.llm_timeout_seconds,
                temperature=0.0,
            )
        self.llm = llm

    def answer(
        self,
        question: str,
        history: list[dict],
        on_status: StatusCallback | None = None,
        top_k: int | None = None,
    ) -> PipelineResult:
        """Retrieve once, synthesize once.

        `top_k` overrides `retrieval_top_k` for this request only; `None`
        keeps the configured default. The first three parameters stay
        positional-compatible with `AgenticPipeline.answer`, which
        `app.api.openai_compat` calls positionally.
        """
        notify = on_status or (lambda _msg: None)
        notify("Retrieving documents…")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            logger.info("no chunks retrieved for question %r", question[:80])
            return PipelineResult(answer=NO_CONTEXT_ANSWER, retrieval_attempts=1)
        notify("Synthesizing answer…")
        prompt = get_prompt_manager().get(
            "synthesizer", context=format_context(chunks), question=question, feedback=""
        )
        answer = self.llm.complete(prompt).text.strip()
        return PipelineResult(
            answer=answer,
            citations=build_citations(chunks),
            chunks=chunks,
            retrieval_attempts=1,
            generation_attempts=1,
        )
