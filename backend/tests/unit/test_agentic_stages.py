"""Stage-layer tests: the search tool and the prompt wiring of each stage.

No CrewAI crew is ever kicked off here -- `CrewStages._kickoff` is patched, so
these run with no Ollama, no Postgres and no Phoenix. What they protect:

* the tool really appends to the *caller's* buffer (pydantic v2 re-validates
  a `list[...]`-annotated field into a brand-new list, which would silently
  break the researcher/tool hand-off -- see `DocumentSearchTool.buffer`);
* every stage renders its prompt with a full set of placeholder values, since
  `PromptManager.get()` raises on any placeholder left unsubstituted.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.pipelines.types import RetrievedChunk

# See tests/unit/test_simple_pipeline.py: pytest's cwd is backend/, so the
# repo-root prompts/ directory has to be pointed at explicitly.
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@pytest.fixture(autouse=True)
def _prompts_dir(monkeypatch):
    monkeypatch.setenv("DOCUMIND_PROMPTS_DIR", str(PROMPTS_DIR))


def _chunk(text="Total: 100", title="Doc", section="A"):
    return RetrievedChunk(text=text, score=0.9, doc_id="d", title=title,
                          section_path=section, pages=[1])


def _retriever(*chunks):
    r = MagicMock()
    r.retrieve.return_value = list(chunks)
    return r


def _stages(retriever=None, outputs=None):
    """Build CrewStages with `_kickoff` stubbed; returns (stages, calls)."""
    from app.agents.stages import CrewStages

    stages = CrewStages(llm=MagicMock(), retriever=retriever or _retriever())
    calls: list[dict] = []
    queue = list(outputs or [])

    def fake_kickoff(role, goal, description, expected_output, tools=None):
        calls.append({"role": role, "description": description, "tools": tools or []})
        return queue.pop(0) if queue else ""

    stages._kickoff = fake_kickoff  # type: ignore[method-assign]
    return stages, calls


# --- DocumentSearchTool ---


def test_tool_appends_to_the_callers_buffer():
    from app.agents.tools import DocumentSearchTool

    buffer: list[RetrievedChunk] = []
    tool = DocumentSearchTool(retriever=_retriever(_chunk("a"), _chunk("b")), buffer=buffer)
    out = tool.run(query="totals")
    assert tool.buffer is buffer, "buffer must be shared with the caller, not copied"
    assert [c.text for c in buffer] == ["a", "b"]
    assert out.startswith("[0] a") and "[1] b" in out


def test_tool_reports_empty_results():
    from app.agents.tools import DocumentSearchTool

    tool = DocumentSearchTool(retriever=_retriever(), buffer=[])
    assert tool.run(query="nothing") == "No results found."


# --- stages ---


def test_route_parses_model_output():
    stages, calls = _stages(outputs=["direct"])
    assert stages.route("hello there") == "direct"
    assert "hello there" in calls[0]["description"]


def test_rewrite_threads_feedback_and_falls_back_to_the_question():
    stages, calls = _stages(outputs=["not json at all"])
    assert stages.rewrite("what is the total?", "user: hi", "too narrow") == [
        "what is the total?"
    ]
    description = calls[0]["description"]
    assert "user: hi" in description and "too narrow" in description


def test_rewrite_with_empty_history_still_renders():
    stages, calls = _stages(outputs=['["a"]'])
    assert stages.rewrite("q", "", "") == ["a"]
    assert "{history}" not in calls[0]["description"]


def test_research_falls_back_to_direct_retrieval_when_tool_unused():
    retriever = _retriever(_chunk("x"), _chunk("x"), _chunk("y"))
    stages, calls = _stages(retriever=retriever, outputs=["searched"])
    chunks = stages.research(["q1"])
    assert [c.text for c in chunks] == ["x", "y"]  # deduplicated by text
    assert calls[0]["tools"], "researcher must be given the document_search tool"


def test_research_survives_a_crew_failure():
    from app.agents.stages import CrewStages

    retriever = _retriever(_chunk("x"))
    stages = CrewStages(llm=MagicMock(), retriever=retriever)

    def boom(*args, **kwargs):
        raise RuntimeError("crew exploded")

    stages._kickoff = boom  # type: ignore[method-assign]
    assert [c.text for c in stages.research(["q1"])] == ["x"]


def test_grade_numbers_chunks_and_parses_indices():
    stages, calls = _stages(outputs=["[1]"])
    assert stages.grade("q", [_chunk("first"), _chunk("second")]) == [1]
    assert "[0] first" in calls[0]["description"]
    assert "[1] second" in calls[0]["description"]


def test_synthesize_includes_context_and_feedback():
    stages, calls = _stages(outputs=["  the total is 100 [Doc, A]  "])
    answer = stages.synthesize("q", [_chunk("Total: 100")], "not grounded")
    assert answer == "the total is 100 [Doc, A]"
    assert "Total: 100" in calls[0]["description"]
    assert "not grounded" in calls[0]["description"]


def test_check_parses_verdict():
    stages, calls = _stages(outputs=["no"])
    assert stages.check("made up", [_chunk()]) is False
    assert "made up" in calls[0]["description"]


def test_direct_answer_is_stripped():
    stages, _ = _stages(outputs=["  hi there  "])
    assert stages.direct_answer("hello") == "hi there"
