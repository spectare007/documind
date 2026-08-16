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
    stages = _stages(grades=[[], []])  # grader rejects everything, twice
    result = _pipeline(stages).answer("q", history=[])
    assert stages.rewrite.call_count == 2          # exactly one retry
    assert "couldn't find" in result.answer.lower()
    assert result.retrieval_attempts == 2
    # second rewrite got corrective feedback
    assert stages.rewrite.call_args_list[1].args[2] != ""


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


def test_failing_check_keeps_the_answer_but_reports_grounded_none():
    """Finding D plus B2. Two things have to hold at once:

    * a verifier crash must not throw away an answer already in hand, and
      must not burn the regeneration attempt as if it were a "no" verdict;
    * the result must NOT claim `grounded=True`. `True` is supposed to mean
      "a checker ran and did not object"; reporting it for a check that never
      ran makes an unchecked answer indistinguishable from a checked one.
    """
    stages = _stages()
    stages.check.side_effect = RuntimeError("verifier blew up")
    result = _pipeline(stages).answer("q", history=[])
    assert result.answer.startswith("answer")
    assert result.grounded is None          # unchecked, not "grounded"
    assert result.grounded is not True
    assert result.generation_attempts == 1  # not retried as if ungrounded


def test_unexpected_stage_failure_returns_a_result_not_an_exception():
    """Finding D: an unexpected stage exception used to escape as a bare 500,
    and in the streaming/chat layer would abort a stream mid-flight with no
    message.
    """
    stages = _stages()
    stages.synthesize.side_effect = RuntimeError("boom")
    result = _pipeline(stages).answer("q", history=[])
    assert result.grounded is None
    assert result.answer and "couldn't" in result.answer.lower()
    assert result.route == "rag"


def test_llm_unavailable_still_propagates_for_the_503_contract():
    """Containment must not swallow the connectivity failures that
    `app.api.query` deliberately maps to a 503. Those are an expected
    operating condition with a designed response, not "unexpected".
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


def test_top_k_reaches_both_retrieval_and_grading():
    """S6: `top_k` used to be accepted by the API, exported into the schema,
    and never read. It must reach the stages that actually size retrieval.
    """
    stages = _stages()
    _pipeline(stages).answer("q", history=[], top_k=11)
    assert stages.research.call_args.args[1] == 11
    assert stages.grade.call_args.args[2] == 11


def test_top_k_defaults_to_none_so_settings_win():
    stages = _stages()
    _pipeline(stages).answer("q", history=[])
    assert stages.research.call_args.args[1] is None
    assert stages.grade.call_args.args[2] is None


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
