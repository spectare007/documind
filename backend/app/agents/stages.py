"""The corrective-RAG stages: one single-agent CrewAI crew per *deciding* stage.

Four roles are CrewAI agents because each one makes a judgment a model has to
make: Router (retrieve or reply directly), Query Rewriter (what to search
for), Answer Synthesizer (what the answer is), Groundedness Checker (whether
the answer is supported). Two stages are plain deterministic code: `research`
(the queries are already chosen, so every one of them is simply searched) and
`grade` (a binary classifier that measurably got *worse* inside an agent
wrapper). Both have docstrings recording why.

Why one crew per stage rather than one multi-agent crew: the pipeline needs
*programmatic* control over the retry loops (bounded attempts, feedback
threaded into the next attempt, an honest `grounded` flag on the way out).
Delegation inside a single crew would hide that control flow inside the LLM's
own judgment and make the attempt counters unbounded and untestable. So
`app.pipelines.agentic.AgenticPipeline` owns the graph, and this module owns
"turn one prompt into one parsed value".

Everything an LLM returns is funnelled through the pure `parse_*` helpers
below. They are the safety layer for a small local model that will sometimes
answer off-format, and each one fails toward *more* work rather than less:
route -> retrieve, ungradeable chunks -> keep them, unreadable verdict ->
treat the answer as grounded rather than block it.
"""

import json
import logging
import re

from crewai import Agent, Crew, Process, Task

from app.agents.llm import get_crew_llm
from app.core.config import get_settings
from app.observability.prompts import get_prompt_manager
from app.pipelines.simple import format_context
from app.pipelines.types import RetrievedChunk
from app.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Agents are told what they are, not what to say: the actual instructions all
# come from the versioned prompt templates via `get_prompt_manager()`.
_BACKSTORY = "You work inside DocuMind, a document search platform."

# A stage agent gets at most this many reasoning iterations before CrewAI
# forces it to answer. Every remaining agent role answers in one, so this is a
# safety cap against a model that talks itself into a loop, not a budget any
# stage is expected to spend.
_MAX_ITER = 3

MAX_REWRITTEN_QUERIES = 3


# --- pure parse helpers (tested directly) ---


def parse_route(text: str) -> str:
    """`"direct"` only on an explicit direct signal; anything else retrieves.

    Anchored to the start of the reply rather than a substring search: the
    router is asked for one word, so a model that answers in a sentence
    ("this is not a direct question") must not flip the route away from
    retrieval on the strength of a passing mention. `\\b` also stops
    "indirect" from matching.
    """
    return "direct" if re.match(r"\W*direct\b", text.strip().lower()) else "rag"


def _extract_json_array(text: str) -> list | None:
    """First JSON array embedded anywhere in `text`, or None.

    Small models like to wrap answers in prose ("Sure, here you go: [...]"),
    so the array is located by regex rather than parsing the whole reply.
    """
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def parse_queries(text: str, fallback: str = "") -> list[str]:
    """Up to `MAX_REWRITTEN_QUERIES` search queries; `fallback` if unparseable."""
    arr = _extract_json_array(text)
    queries = [q.strip() for q in (arr or []) if isinstance(q, str) and q.strip()]
    if queries:
        return queries[:MAX_REWRITTEN_QUERIES]
    return [fallback] if fallback else []


def parse_verdict(text: str) -> bool:
    """Groundedness verdict, failing open: only an explicit "no" blocks."""
    return not text.strip().lower().startswith("no")


def parse_relevance_verdict(text: str) -> bool:
    """Grader verdict: an explicit "yes" keeps the chunk, "no" drops it, and
    anything else -- an off-format reply, an empty string -- fails open
    (kept), same fail-open philosophy as every parser in this module.

    Deliberately its own function rather than a reuse of `parse_verdict`.
    An earlier iteration (see `grade()`'s docstring) asked the
    grader for RELEVANT/IRRELEVANT instead of yes/no, to dodge a CrewAI
    agent-framing bug, and reused `parse_verdict`'s "does the reply start
    with an explicit no" check against that vocabulary. That was silently
    *inverted*: "IRRELEVANT" does not start with "no", so the fail-open
    logic kept every chunk regardless of what the model actually judged --
    the chosen vocabulary meant the opposite of what the parser tested for,
    and the grader became a no-op rubber stamp. The lesson generalizes
    beyond that one word pair: a verdict parser must be checked against its
    own vocabulary, not assumed compatible with a check written for a
    different one. This function keys on an affirmative "yes" instead (the
    polarity grading actually needs -- an unreadable verdict must not
    narrow the evidence the synthesizer sees), and, like `parse_route`, is
    anchored with `startswith` rather than a substring search: the model is
    asked for one word, so a stray "yes" or "no" appearing later in a
    longer off-format reply must not flip the result.
    """
    verdict = text.strip().lower()
    if verdict.startswith("yes"):
        return True
    if verdict.startswith("no"):
        return False
    return True


# --- crew stages: one single-agent Crew per deciding pipeline step ---


class CrewStages:
    """The six stages of the corrective-RAG graph, plus the direct reply.

    Four are CrewAI agents (`route`, `rewrite`, `synthesize`, `check`, and
    `direct_answer` on the no-retrieval branch); `research` and `grade` are
    plain code. See the module docstring.

    Constructor arguments exist for tests and for the evaluation harness:
    passing an `llm`/`retriever` avoids touching Ollama or Postgres.
    """

    def __init__(self, llm=None, retriever: HybridRetriever | None = None) -> None:
        self.llm = llm or get_crew_llm()
        self.retriever = retriever or HybridRetriever()
        self.prompts = get_prompt_manager()
        self.settings = get_settings()

    def _kickoff(self, role: str, goal: str, description: str,
                 expected_output: str, tools: list | None = None) -> str:
        """Run a one-agent, one-task crew and return its raw text output.

        `str(CrewOutput)` yields `.raw` (the final task's text) unless a
        structured output was requested, which it never is here.
        """
        agent = Agent(role=role, goal=goal, backstory=_BACKSTORY,
                      llm=self.llm, tools=tools or [], verbose=False,
                      allow_delegation=False, max_iter=_MAX_ITER)
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return str(crew.kickoff())

    def route(self, question: str) -> str:
        out = self._kickoff("Query Router", "Classify queries precisely",
                            self.prompts.get("router", question=question),
                            "one word: rag or direct")
        return parse_route(out)

    def rewrite(self, question: str, history: str, feedback: str) -> list[str]:
        fb = f"\nFeedback from a failed retrieval attempt: {feedback}" if feedback else ""
        out = self._kickoff(
            "Query Rewriter", "Produce excellent standalone search queries",
            self.prompts.get("rewriter", question=question,
                             history=history or "(empty)", feedback=fb),
            "JSON array of 1-3 query strings",
        )
        return parse_queries(out, fallback=question)

    def research(self, queries: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
        """Search once per query and return the deduplicated chunks.

        Deliberately **not** a CrewAI agent, unlike Router, Rewriter,
        Synthesizer and Checker. It used to be one, and the agent framing was
        pure overhead: the stage wrapped an `Agent`+`Task`+`Crew` around a
        `document_search` tool, then discarded the agent's returned text
        entirely and read the structured chunks back out of a buffer the tool
        wrote to. There is no decision left for a model to make at this point
        -- the Rewriter has already chosen the queries, and every one of them
        is searched. So the completion bought nothing and cost a full LLM
        round trip (plus its own failure modes: a small model that answers
        from memory without calling the tool at all, which needed a direct
        retrieval fallback, which then needed a usage-count guard to stop it
        double-retrieving) on a path whose measured median latency is about
        two minutes. Retrieval and grading are now the two deterministic
        steps in the graph; the four roles that genuinely make a judgment
        stay agents.

        `top_k` is the per-request override for `retrieval_top_k`; `None`
        keeps the configured default.
        """
        chunks: list[RetrievedChunk] = []
        for query in queries:
            chunks.extend(self.retriever.retrieve(query, top_k=top_k))
        # Deduplicate on (doc_id, text): identical text in two different
        # documents is two real citations, and collapsing on text alone
        # silently dropped one of them.
        seen: set[tuple[str, str]] = set()
        unique = [c for c in chunks
                  if not ((c.doc_id, c.text) in seen or seen.add((c.doc_id, c.text)))]
        logger.info("research stage: %d queries -> %d unique chunks", len(queries), len(unique))
        return unique

    def grade(self, question: str, chunks: list[RetrievedChunk],
              top_k: int | None = None) -> list[int]:
        """Indices of the chunks worth answering from -- one binary verdict
        per chunk, single-token output, called as a *direct* LLM completion
        rather than through a CrewAI Agent/Task like every other stage.

        Direct LLM completion, not a CrewAI `Agent`/`Task` like every other
        stage -- three iterations got here, measured live against qwen2.5:3b:

        1. JSON-array form ("reply with [0, 2]") is unusable once CrewAI
           appends its expected-output boilerplate: the model answered `[]`
           to every input regardless of relevance.
        2. A yes/no rewording, still run through `_kickoff()`
           (`Agent`+`Task`+`Crew`), didn't fix it either. The cause was
           CrewAI's own scaffolding: its injected system message (role+goal)
           alone flips this model negative independent of content, even on
           the control question "Is grass green? Answer YES or NO." Swapping
           the vocabulary to RELEVANT/IRRELEVANT traded that for a different
           failure -- the model settled on "IRRELEVANT" for everything, which
           doesn't start with "no", so the fail-open parser kept every chunk:
           a rubber-stamp grader wearing a discriminating one's clothes (see
           `parse_relevance_verdict`).
        3. Removing the Agent/Task wrapper -- a direct `self.llm.call(...)`,
           no system message, no expected-output footer -- restored genuine
           per-chunk discrimination (6/6 correct live, both directions).
           Grading is a binary classifier with no goal or tools, so the agent
           framing bought nothing here; the other five roles are unaffected.

        Bounded on purpose: at most `retrieval_top_k` chunks are graded (or
        `top_k` if given), highest-scoring first, so a broad retrieval can't
        turn one request into a dozen sequential CPU completions.

        Fails open per chunk via `parse_relevance_verdict`: only an explicit
        "yes" keeps a chunk, and a raised exception (LLM call failed) also
        keeps it. An empty result therefore means every graded chunk was
        explicitly rejected -- the real judgment that drives the corrective
        retrieval retry -- never an artifact of an off-format reply or a
        transient failure.
        """
        if not chunks:
            return []
        limit = top_k or self.settings.retrieval_top_k
        ranked = sorted(range(len(chunks)), key=lambda i: chunks[i].score, reverse=True)
        graded = ranked[:limit]
        if len(ranked) > limit:
            logger.info(
                "grading top %d of %d chunks by score (limit=%d)",
                len(graded), len(chunks), limit,
            )

        keep: list[int] = []
        for i in graded:
            prompt = self.prompts.get("grader", question=question, chunk=chunks[i].text)
            try:
                out = self.llm.call([{"role": "user", "content": prompt}])
            except Exception as exc:
                logger.warning("grading chunk %d failed (%s); keeping it", i, exc)
                keep.append(i)
                continue
            if parse_relevance_verdict(out):
                keep.append(i)
        logger.info("grade stage: kept %d of %d graded chunk(s)", len(keep), len(graded))
        return sorted(keep)

    def synthesize(self, question: str, chunks: list[RetrievedChunk], feedback: str) -> str:
        fb = f"\nIMPORTANT: {feedback}" if feedback else ""
        return self._kickoff(
            "Answer Synthesizer", "Write grounded, cited answers",
            self.prompts.get("synthesizer", context=format_context(chunks),
                             question=question, feedback=fb),
            "a grounded answer with [title, section] citations",
        ).strip()

    def check(self, answer: str, chunks: list[RetrievedChunk]) -> bool:
        out = self._kickoff(
            "Groundedness Checker", "Detect unsupported claims",
            self.prompts.get("hallucination_checker",
                             context=format_context(chunks), answer=answer),
            "one word: yes or no",
        )
        return parse_verdict(out)

    def direct_answer(self, question: str) -> str:
        """Reply to small talk / questions about the assistant, no retrieval.

        Deliberately the one task description built in code rather than read
        from `prompts/`: it is a chit-chat instruction with nothing to tune or
        evaluate, and the five versioned prompt templates are a fixed, synced
        set (`app.observability.prompts.PROMPT_NAMES`) covering the retrieval
        graph. If this ever needs tuning it should become a sixth template.
        """
        return self._kickoff(
            "Assistant", "Answer briefly and helpfully",
            f"You are DocuMind, a document search assistant. Reply briefly to: {question}",
            "a short friendly reply",
        ).strip()
