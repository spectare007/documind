"""Corrective-RAG orchestration: route -> (rewrite -> research -> grade)* ->
(synthesize -> check)*.

This module owns the control flow; `app.agents.stages.CrewStages` owns the
LLM calls. Keeping them apart is what makes the loops testable without a
model (tests/unit/test_agentic_orchestration.py) and what keeps the retry
budget a hard, configured number rather than something an agent decides.

THREE INDEPENDENT WAYS THIS CANNOT RUN AWAY OR FAIL CLOSED:

1. *Attempt bounds.* Both corrective loops are bounded by settings
   (`max_retrieval_attempts`, `max_generation_attempts`, both 2): one attempt
   plus at most one correction. Exhausting retrieval returns the honest
   "nothing found" answer; exhausting generation still returns the answer
   with `grounded=False`, so the caller can flag it rather than show nothing.

2. *A wall-clock budget* (`request_budget_seconds`, default 300). Attempt
   bounds alone do not bound *time*: `llm_timeout_seconds` caps a single
   completion, but one request makes up to 11 of them, so a degraded Ollama
   could hold a worker for ~25 minutes. The budget is checked at stage
   boundaries and stops the pipeline starting more work, returning the best
   result it already has. It deliberately cannot cut short a completion
   already in flight -- that is `llm_timeout_seconds`' job -- and it never
   skips the *first* attempt of either loop, because a request that returns
   without trying anything is worse than a slow one.

3. *Exception containment.* Any unexpected stage failure returns a
   `PipelineResult` carrying a friendly message and `grounded=None` instead
   of propagating: an escaped exception is a bare HTTP 500 today, and in the
   streaming layer (Task 11) it would abort a response mid-flight with
   nothing written. The one deliberate exception is
   `LLM_UNAVAILABLE_ERRORS`, which is re-raised so `app.api.query` can still
   turn it into a structured 503 -- upstream being down is an expected
   condition with a designed response, not an unexpected failure. `check()`
   gets its own inner guard so a verifier crash can never discard an answer
   that is already in hand.
"""

import logging
import time

from app.core.config import get_settings
from app.core.errors import LLM_UNAVAILABLE_ERRORS
from app.pipelines.simple import NO_CONTEXT_ANSWER
from app.pipelines.types import PipelineResult, RetrievedChunk, StatusCallback
from app.retrieval.retriever import build_citations

logger = logging.getLogger(__name__)

# How many prior turns of chat are handed to the rewriter for pronoun
# resolution. Enough for "and the previous one?" chains, small enough not to
# crowd a 3B model's context with stale topics.
HISTORY_TURNS = 6

# User-facing copy for the two "we stopped early" outcomes. Both are written
# to be honest about what happened without leaking internals.
BUDGET_EXCEEDED_ANSWER = (
    "That question took too long to answer, so I stopped before I had a "
    "complete result. Please try again, or ask something more specific."
)
FAILURE_ANSWER = (
    "Sorry, I couldn't complete that request -- something went wrong while I "
    "was working on it. Please try again."
)

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
        """Answer `question`. Never raises except for upstream unavailability.

        See the module docstring for the three independent bounds. The
        signature is fixed: Task 11 calls this positionally.
        """
        try:
            return self._answer(question, history, on_status)
        except LLM_UNAVAILABLE_ERRORS:
            # Expected operating condition with a designed response (503).
            # Re-raised deliberately -- see `app.core.errors`.
            raise
        except Exception:
            logger.exception("agentic pipeline failed for question %r", question[:120])
            return PipelineResult(answer=FAILURE_ANSWER)

    def _answer(self, question: str, history: list[dict],
                on_status: StatusCallback | None) -> PipelineResult:
        notify = on_status or (lambda _msg: None)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:]
        )
        deadline = time.monotonic() + self.settings.request_budget_seconds

        def out_of_time(stage: str) -> bool:
            if time.monotonic() < deadline:
                return False
            logger.warning(
                "request budget of %.0fs exhausted before %s; returning best "
                "result so far", self.settings.request_budget_seconds, stage,
            )
            return True

        notify("Routing query…")
        if self.stages.route(question) == "direct":
            logger.info("routed direct (no retrieval) for %r", question[:80])
            return PipelineResult(answer=self.stages.direct_answer(question), route="direct")

        relevant: list[RetrievedChunk] = []
        feedback = ""
        attempts = 0
        budget_hit = False
        for attempt in range(self.settings.max_retrieval_attempts):
            # The first attempt always runs: returning without having tried
            # to retrieve anything is worse than being slow.
            if attempt > 0 and out_of_time("retrieval retry"):
                budget_hit = True
                break
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
            return PipelineResult(
                answer=BUDGET_EXCEEDED_ANSWER if budget_hit else NO_CONTEXT_ANSWER,
                retrieval_attempts=attempts,
            )

        grounded, answer, gen_attempts = True, "", 0
        feedback = ""
        for attempt in range(self.settings.max_generation_attempts):
            if attempt > 0 and out_of_time("answer regeneration"):
                break
            gen_attempts = attempt + 1
            notify("Synthesizing answer…")
            answer = self.stages.synthesize(question, relevant, feedback)
            notify("Verifying groundedness…")
            grounded = self._check(answer, relevant)
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

    def _check(self, answer: str, chunks: list[RetrievedChunk]) -> bool:
        """Groundedness check that fails open, matching `parse_verdict`.

        By this point an answer exists. A verifier that crashes is not
        evidence the answer is bad, and treating it as "ungrounded" would
        either burn the regeneration attempt or (on the last attempt) ship a
        good answer flagged as untrustworthy. Both are worse than trusting
        it. Catches broadly on purpose -- including upstream errors, since
        one flaky verifier call should not cost the user an answer we
        already have.
        """
        try:
            return self.stages.check(answer, chunks)
        except Exception as exc:
            logger.warning("groundedness check failed (%s); assuming grounded", exc)
            return True
