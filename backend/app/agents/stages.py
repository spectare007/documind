"""One single-agent CrewAI crew per corrective-RAG stage.

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
from app.agents.tools import DocumentSearchTool
from app.core.config import get_settings
from app.observability.prompts import get_prompt_manager
from app.pipelines.simple import format_context
from app.pipelines.types import RetrievedChunk
from app.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Agents are told what they are, not what to say: the actual instructions all
# come from the versioned prompt templates via `get_prompt_manager()`.
_BACKSTORY = "You work inside DocuMind, a document search platform."

# A stage agent gets at most this many reasoning/tool iterations before CrewAI
# forces it to answer. Only the researcher genuinely loops (one tool call per
# rewritten query, of which there are at most three); the rest answer in one.
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


def parse_indices(text: str, n_chunks: int) -> list[int]:
    """Relevant chunk indices from a JSON-array reply, failing open.

    NOTE: `grade()` no longer calls this -- it now asks for one binary
    verdict per chunk (see its docstring for the measurements that forced
    that change) and uses `parse_relevance_verdict`. This stays as a public, tested
    parse helper for any caller that does get an array-of-indices reply
    (e.g. Task 12's evaluation harness scoring a larger model, which handles
    the array form fine); it is the only parser here that reads one.

    Three cases, and the distinction between the last two matters:

    * no JSON array at all -> keep every chunk. A grader that cannot answer
      in format must not be able to starve the synthesizer.
    * an explicit empty array -> keep nothing. This is a real judgment and is
      respected: it is what triggers the corrective retrieval retry.
    * a non-empty array whose entries are *all* out of range -> keep every
      chunk. This is a malformed reply wearing the right punctuation, not a
      judgment of irrelevance, so it gets the same fail-open treatment as an
      unparseable one. Observed live: given one chunk, qwen2.5:3b copies the
      `e.g. [0, 2]` example out of the grader prompt and answers `[2]`, which
      would otherwise silently reduce to "nothing is relevant" and make the
      whole pipeline answer "I couldn't find anything" for a corpus that does
      contain the answer.

    Mixed replies (`[0, 9]` over 3 chunks) keep the valid entries and drop the
    rest -- there the in-range index is real signal.
    """
    arr = _extract_json_array(text)
    if arr is None:
        return list(range(n_chunks))
    indices = sorted(
        {int(i) for i in arr if isinstance(i, (int, float)) and 0 <= int(i) < n_chunks}
    )
    if arr and not indices:
        logger.warning(
            "grader returned %r: no index in range 0..%d; keeping all chunks",
            arr, n_chunks - 1,
        )
        return list(range(n_chunks))
    return indices


def parse_verdict(text: str) -> bool:
    """Groundedness verdict, failing open: only an explicit "no" blocks."""
    return not text.strip().lower().startswith("no")


def parse_relevance_verdict(text: str) -> bool:
    """Grader verdict: an explicit "yes" keeps the chunk, "no" drops it, and
    anything else -- an off-format reply, an empty string -- fails open
    (kept), same fail-open philosophy as every parser in this module.

    Deliberately its own function rather than a reuse of `parse_verdict`.
    An earlier iteration (see `grade()`'s docstring and ADR-9) asked the
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


# --- crew stages: one single-agent Crew per pipeline step ---


class CrewStages:
    """The six agent roles of the corrective-RAG graph, plus the direct reply.

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

    def research(self, queries: list[str]) -> list[RetrievedChunk]:
        """Search once per query and return the deduplicated chunks.

        The agent's own reply is thrown away -- only the structured chunks it
        pushed into `buffer` matter. Two safety nets keep this stage from ever
        returning nothing for a recoverable reason: a crew failure is logged
        and swallowed, and an agent that never called the tool (small models
        sometimes just answer from memory) is backstopped by running the
        retriever directly.
        """
        buffer: list[RetrievedChunk] = []
        tool = DocumentSearchTool(retriever=self.retriever, buffer=buffer)
        description = (
            "Use the document_search tool once per query to gather evidence. Queries:\n"
            + "\n".join(f"- {q}" for q in queries)
            + "\nAfter searching, reply with a one-line summary of what was found."
        )
        try:
            self._kickoff("Researcher", "Gather relevant document evidence",
                          description, "one-line summary", tools=[tool])
        except Exception as exc:
            logger.warning("researcher crew failed (%s); falling back to direct retrieval", exc)
        # `current_usage_count` is incremented by CrewAI's `BaseTool.run`
        # *before* it delegates to `_run`, so 0 means "the tool was never
        # invoked" and nothing else. Guarding on `not buffer` instead
        # conflated that with "the agent searched and found nothing", and so
        # re-ran every query through the retriever a second time on any
        # empty-result question -- doubling the embedding calls for the most
        # common failure case. A crew that died before its first tool call
        # still has count 0, so the crew-failure fallback keeps working.
        if tool.current_usage_count == 0:
            logger.info("researcher never called document_search; retrieving directly")
            for q in queries:
                buffer.extend(self.retriever.retrieve(q))
        # Deduplicate on (doc_id, text): identical text in two different
        # documents is two real citations, and collapsing on text alone
        # silently dropped one of them.
        seen: set[tuple[str, str]] = set()
        unique = [c for c in buffer
                  if not ((c.doc_id, c.text) in seen or seen.add((c.doc_id, c.text)))]
        logger.info("research stage: %d queries -> %d unique chunks", len(queries), len(unique))
        return unique

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> list[int]:
        """Indices of the chunks worth answering from -- one binary verdict
        per chunk, single-token output, called as a *direct* LLM completion
        rather than through a CrewAI Agent/Task like every other stage.

        Three iterations got here, each one measured live against the
        deployed qwen2.5:3b, not assumed:

        1. JSON-array form ("reply with [0, 2]") is unusable on a 3B model
           once CrewAI appends its expected-output boilerplate to the task:
           qwen2.5:3b answered `[]` to *every* input -- relevant chunk,
           irrelevant chunk, no chunk -- which made the whole pipeline say
           "I couldn't find anything" for a corpus that did contain the
           answer.
        2. A rewording to one yes/no verdict per chunk, still run through
           the same `_kickoff()` (`Agent`+`Task`+`Crew`) path as every other
           stage, did not fix it: it still answered "no" for essentially
           every chunk regardless of content. The cause was not the
           question's wording -- it was CrewAI's own scaffolding. Isolating
           the variables live showed the injected system message
           (role+goal) alone flips this model negative independent of
           content: even the control question "Is grass green? Answer YES
           or NO." answered "no" once that system message was present, and
           "yes" with it removed. Sidestepping the vocabulary (asking for
           RELEVANT/IRRELEVANT instead of yes/no) only produced a different
           failure: the model settled on "IRRELEVANT" for every chunk, which
           doesn't start with "no", so the fail-open parser kept everything
           -- a rubber-stamp grader, not a discriminating one, and an
           inverted-semantics near miss (see `parse_relevance_verdict`).
        3. Removing the CrewAI Agent/Task wrapper entirely for this one
           stage -- a direct `self.llm.call(...)` completion, no system
           message, no expected-output footer -- restored genuine per-chunk
           discrimination: 6/6 correct live (both true-positive and
           true-negative), including on a genuinely unanswerable question.
           Grading is a binary classifier with no goal, no tools, no
           delegation and no multi-step reasoning, so an agent framing was
           never buying anything here; the other five roles keep using
           `_kickoff()` and the multi-agent architecture is otherwise
           unchanged. See ADR-9 in `doc/design-decisions.md` for the full
           record, deliberately including the near-miss.

        Bounded on purpose: at most `retrieval_top_k` chunks are graded,
        highest-scoring first, so a broad retrieval cannot turn one request
        into a dozen sequential CPU completions. Returned indices always
        address the *caller's* list, so the signature and semantics are
        unchanged from the array version.

        Fails open per chunk via `parse_relevance_verdict`: only an explicit
        "yes" keeps a chunk that the parser can read, and a raised exception
        (LLM call failed) also keeps it rather than silently dropping it --
        see that function's docstring for why this needed its own parser
        instead of reusing `parse_verdict`. An empty result therefore means
        every graded chunk was explicitly rejected -- a real judgment, which
        is what drives the corrective retrieval retry -- and can never be an
        artifact of a model answering off-format or a transient LLM failure.
        """
        if not chunks:
            return []
        limit = self.settings.retrieval_top_k
        ranked = sorted(range(len(chunks)), key=lambda i: chunks[i].score, reverse=True)
        graded = ranked[:limit]
        if len(ranked) > limit:
            logger.info(
                "grading top %d of %d chunks by score (retrieval_top_k=%d)",
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
