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
