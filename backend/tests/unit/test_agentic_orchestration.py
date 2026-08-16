"""Orchestration contract for the corrective-RAG pipeline.

The stage layer (CrewAI crews, and therefore the LLM) is mocked out entirely:
these tests assert *control flow only* -- routing short-circuits, both
corrective loops are hard-bounded to two attempts, feedback is threaded into
the retry, and an ungrounded answer still ships with an honest flag.
"""

from unittest.mock import MagicMock

from app.pipelines.types import RetrievedChunk


def _chunk(text="c1"):
    return RetrievedChunk(text=text, score=0.9, doc_id="d", title="Doc", section_path="S", pages=[1])


def _stages(route="rag", grades=None, verdicts=None, chunks=None):
    s = MagicMock()
    s.route.return_value = route
    s.rewrite.return_value = ["q1"]
    s.research.return_value = chunks if chunks is not None else [_chunk()]
    s.grade.side_effect = grades or [[0]]
    s.check.side_effect = verdicts or [True]
    s.synthesize.return_value = "answer [Doc, S]"
    s.direct_answer.return_value = "hi!"
    return s


def _pipeline(stages):
    from app.pipelines.agentic import AgenticPipeline
    return AgenticPipeline(stages=stages)


def test_direct_route_skips_retrieval():
    stages = _stages(route="direct")
    result = _pipeline(stages).answer("hello", history=[])
    assert result.route == "direct" and result.answer == "hi!"
    stages.rewrite.assert_not_called()


def test_happy_path_single_pass():
    stages = _stages()
    result = _pipeline(stages).answer("q", history=[])
    assert result.answer.startswith("answer")
    assert result.grounded is True
    assert result.retrieval_attempts == 1 and result.generation_attempts == 1
    assert result.citations and result.citations[0].title == "Doc"


def test_corrective_loop_bounded_to_two_attempts():
    """The corrective retry is still exactly one attempt, and still bounded.

    NOTE: prior to the grader fail-open fix, a grader that rejected every
    chunk on both attempts made the pipeline refuse ("couldn't find"). That
    was the bug: retrieval *had* found real, ranked chunks, and a rejecting
    3B classifier discarded them all. The fix changes this exact scenario's
    outcome to a fallback answer -- see
    test_grader_rejects_everything_falls_back_to_top_retrieved below for
    that contract. This test now only asserts what's unchanged: the retry is
    still attempted exactly once, feedback is still threaded into it, and
    the attempt count is still reported accurately.
    """
    stages = _stages(grades=[[], []])  # grader rejects everything, twice
    result = _pipeline(stages).answer("q", history=[])
    assert stages.rewrite.call_count == 2          # exactly one retry
    assert result.retrieval_attempts == 2
    # second rewrite got corrective feedback
    assert stages.rewrite.call_args_list[1].args[2] != ""
    # grader rejected everything both times, but retrieval did return chunks,
    # so this is the fail-open fallback, not a refusal.
    assert result.grader_fallback is True
    assert "couldn't find" not in result.answer.lower()


def test_grader_rejects_everything_falls_back_to_top_retrieved():
    """Finding: the relevance grader (a 3B classifier) rejected every chunk
    on real golden-set questions that retrieval had in fact answered
    correctly in `simple` mode using the same retrieval. Discarding all of
    retrieval's ranked output on the grader's say-so turns answerable
    questions into refusals. When this happens, the pipeline must fail open
    to the top-`retrieval_top_k` retrieved chunks (by retrieval score) and
    still attempt synthesis, flagging `grader_fallback=True` -- not refuse.
    """
    chunks = [
        RetrievedChunk(text="low", score=0.1, doc_id="d", title="Doc", section_path="S"),
        RetrievedChunk(text="high", score=0.9, doc_id="d", title="Doc", section_path="S"),
        RetrievedChunk(text="mid", score=0.5, doc_id="d", title="Doc", section_path="S"),
    ]
    stages = _stages(grades=[[], []], chunks=chunks)  # grader rejects everything
    result = _pipeline(stages).answer("q", history=[])

    assert result.grader_fallback is True
    assert result.answer.startswith("answer")            # synthesis ran, not a refusal
    assert "couldn't find" not in result.answer.lower()
    assert result.citations and result.citations[0].title == "Doc"
    # synthesize was called with the retrieved chunks ranked by score, best first
    synth_chunks = stages.synthesize.call_args_list[-1].args[1]
    assert [c.text for c in synth_chunks] == ["high", "mid", "low"]


def test_grader_fallback_respects_retrieval_top_k(monkeypatch):
    """The fallback must not just dump every retrieved chunk into synthesis
    -- it takes the top N where N is `settings.retrieval_top_k`, same limit
    the grader itself uses, so a broad retrieval can't blow the context
    budget just because the grader rejected everything.
    """
    monkeypatch.setenv("DOCUMIND_RETRIEVAL_TOP_K", "2")
    chunks = [
        RetrievedChunk(text=f"c{i}", score=float(i), doc_id="d", title="Doc", section_path="S")
        for i in range(5)
    ]
    stages = _stages(grades=[[], []], chunks=chunks)
    result = _pipeline(stages).answer("q", history=[])

    assert result.grader_fallback is True
    assert len(result.chunks) == 2
    assert [c.text for c in result.chunks] == ["c4", "c3"]  # highest scores


def test_retrieval_returns_nothing_at_all_refuses_without_fallback():
    """When retrieval genuinely never found anything (not even chunks the
    grader could reject), the refusal path is untouched: there is nothing to
    fail open to.
    """
    stages = _stages(chunks=[])
    result = _pipeline(stages).answer("q", history=[])
    assert "couldn't find" in result.answer.lower()
    assert result.grader_fallback is False
    stages.grade.assert_not_called()
    stages.synthesize.assert_not_called()


def test_grader_keeps_some_chunks_is_the_normal_path_not_a_fallback():
    """When the grader keeps at least one chunk, that's the ordinary path:
    no fallback, and it must not be flagged as one.
    """
    stages = _stages()  # default grade keeps index 0
    result = _pipeline(stages).answer("q", history=[])
    assert result.grader_fallback is False
    assert result.answer.startswith("answer")


def test_regeneration_on_hallucination_bounded():
    stages = _stages(verdicts=[False, False])
    result = _pipeline(stages).answer("q", history=[])
    assert stages.synthesize.call_count == 2       # exactly one regeneration
    assert result.grounded is False                # shipped with honest flag
    assert result.generation_attempts == 2


def test_empty_retrieval_retries_then_gives_up_without_generating():
    stages = _stages(chunks=[])
    result = _pipeline(stages).answer("q", history=[])
    assert stages.research.call_count == 2
    stages.grade.assert_not_called()
    stages.synthesize.assert_not_called()
    assert "couldn't find" in result.answer.lower()
    assert result.retrieval_attempts == 2 and result.generation_attempts == 0
    assert result.grounded is None


def test_failing_check_fails_open_and_keeps_the_answer():
    """Finding D: the worst outcome is `check()` throwing away an answer we
    already have. A verifier failure must not cost the user the answer.
    """
    stages = _stages()
    stages.check.side_effect = RuntimeError("verifier blew up")
    result = _pipeline(stages).answer("q", history=[])
    assert result.answer.startswith("answer")
    assert result.grounded is True          # failed open, same as parse_verdict
    assert result.generation_attempts == 1  # not retried as if ungrounded


def test_unexpected_stage_failure_returns_a_result_not_an_exception():
    """Finding D: an unexpected stage exception used to escape as a bare 500,
    and in Task 11 would abort a stream mid-flight with no message.
    """
    stages = _stages()
    stages.synthesize.side_effect = RuntimeError("boom")
    result = _pipeline(stages).answer("q", history=[])
    assert result.grounded is None
    assert result.answer and "couldn't" in result.answer.lower()
    assert result.route == "rag"


def test_llm_unavailable_still_propagates_for_the_503_contract():
    """Containment must not swallow the connectivity failures that
    `app.api.query` deliberately maps to a 503 (Task 9's fix). Those are an
    expected operating condition with a designed response, not "unexpected".
    """
    import httpx
    import pytest

    from app.core.errors import LLM_UNAVAILABLE_ERRORS

    assert issubclass(httpx.ConnectTimeout, LLM_UNAVAILABLE_ERRORS)
    stages = _stages()
    stages.synthesize.side_effect = httpx.ConnectTimeout("timed out")
    with pytest.raises(httpx.ConnectTimeout):
        _pipeline(stages).answer("q", history=[])


def test_request_budget_stops_the_retrieval_loop(monkeypatch):
    """Finding E: `llm_timeout_seconds` bounds ONE completion, but a request
    makes up to 11 -- a degraded Ollama could hold a worker for ~25 minutes.
    A wall-clock budget must stop the loops and return something.
    """
    monkeypatch.setenv("DOCUMIND_REQUEST_BUDGET_SECONDS", "0")
    stages = _stages(grades=[[], []])
    result = _pipeline(stages).answer("q", history=[])
    assert stages.rewrite.call_count == 1, "budget stops the second attempt"
    assert result.retrieval_attempts == 1
    assert "too long" in result.answer.lower()
    assert result.grounded is None


def test_request_budget_returns_the_answer_it_already_has(monkeypatch):
    """Budget exhaustion mid-generation must ship the answer in hand, not
    discard it -- the same fail-open instinct as `check()`.
    """
    monkeypatch.setenv("DOCUMIND_REQUEST_BUDGET_SECONDS", "0")
    stages = _stages(verdicts=[False, False])
    result = _pipeline(stages).answer("q", history=[])
    assert stages.synthesize.call_count == 1, "budget stops the regeneration"
    assert result.answer.startswith("answer")
    assert result.grounded is False
    assert result.citations and result.citations[0].title == "Doc"


def test_history_is_windowed_and_status_messages_emitted():
    stages = _stages()
    history = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    statuses: list[str] = []
    _pipeline(stages).answer("q", history=history, on_status=statuses.append)
    history_text = stages.rewrite.call_args_list[0].args[1]
    assert "m7" in history_text and "m1" not in history_text  # last 6 turns only
    assert [s.split("…")[0] for s in statuses] == [
        "Routing query", "Rewriting query", "Searching documents",
        "Grading context", "Synthesizing answer", "Verifying groundedness",
    ]
