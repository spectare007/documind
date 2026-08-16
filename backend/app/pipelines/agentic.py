"""Corrective-RAG orchestration: route -> (rewrite -> research -> grade)* ->
(synthesize -> check)*.

This module owns the control flow; `app.agents.stages.CrewStages` owns the
LLM calls. Keeping them apart is what makes the loops testable without a
model (tests/unit/test_agentic_orchestration.py) and what keeps the retry
budget a hard, configured number rather than something an agent decides.

Both corrective loops are bounded by settings (`max_retrieval_attempts`,
`max_generation_attempts`, both 2): one attempt plus at most one correction.
Neither loop can fail closed -- exhausting retrieval returns the honest
"nothing found" answer, and exhausting generation still returns the answer
with `grounded=False` so the caller (and the UI) can flag it rather than
showing the user nothing.
"""

import logging

from app.core.config import get_settings
from app.pipelines.simple import NO_CONTEXT_ANSWER
from app.pipelines.types import PipelineResult, RetrievedChunk, StatusCallback
from app.retrieval.retriever import build_citations

logger = logging.getLogger(__name__)

# How many prior turns of chat are handed to the rewriter for pronoun
# resolution. Enough for "and the previous one?" chains, small enough not to
# crowd a 3B model's context with stale topics.
HISTORY_TURNS = 6

_NO_MATCH_FEEDBACK = "No documents matched; try broader or different terms."
_IRRELEVANT_FEEDBACK = (
    "Retrieved chunks were judged irrelevant; rephrase with more specific terms."
)
_UNGROUNDED_FEEDBACK = (
    "Your previous answer contained claims not supported by the context. "
    "Use only facts from the context."
)


class AgenticPipeline:
    """Corrective RAG: route -> (rewrite -> research -> grade)* -> (synthesize -> check)*."""

    def __init__(self, stages=None) -> None:
        if stages is None:
            # Imported lazily so that constructing the *simple* pipeline, or
            # importing this module in tests, never drags in CrewAI/LiteLLM.
            from app.agents.stages import CrewStages
            stages = CrewStages()
        self.stages = stages
        self.settings = get_settings()

    def answer(self, question: str, history: list[dict],
               on_status: StatusCallback | None = None) -> PipelineResult:
        notify = on_status or (lambda _msg: None)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:]
        )

        notify("Routing query…")
        if self.stages.route(question) == "direct":
            logger.info("routed direct (no retrieval) for %r", question[:80])
            return PipelineResult(answer=self.stages.direct_answer(question), route="direct")

        relevant, feedback = [], ""
        attempts = 0
        for attempt in range(self.settings.max_retrieval_attempts):
            attempts = attempt + 1
            notify("Rewriting query…")
            queries = self.stages.rewrite(question, history_text, feedback)
            notify("Searching documents…")
            all_chunks: list[RetrievedChunk] = self.stages.research(queries)
            if not all_chunks:
                feedback = _NO_MATCH_FEEDBACK
                continue
            notify("Grading context…")
            keep = self.stages.grade(question, all_chunks)
            relevant = [all_chunks[i] for i in keep]
            if relevant:
                break
            feedback = _IRRELEVANT_FEEDBACK

        if not relevant:
            logger.info("no relevant context after %d retrieval attempt(s)", attempts)
            return PipelineResult(answer=NO_CONTEXT_ANSWER, retrieval_attempts=attempts)

        grounded, answer, gen_attempts = True, "", 0
        feedback = ""
        for attempt in range(self.settings.max_generation_attempts):
            gen_attempts = attempt + 1
            notify("Synthesizing answer…")
            answer = self.stages.synthesize(question, relevant, feedback)
            notify("Verifying groundedness…")
            grounded = self.stages.check(answer, relevant)
            if grounded:
                break
            feedback = _UNGROUNDED_FEEDBACK

        if not grounded:
            logger.warning(
                "answer still ungrounded after %d generation attempt(s); "
                "returning it flagged", gen_attempts,
            )
        return PipelineResult(
            answer=answer, citations=build_citations(relevant), chunks=relevant,
            grounded=grounded, route="rag",
            retrieval_attempts=attempts, generation_attempts=gen_attempts,
        )
