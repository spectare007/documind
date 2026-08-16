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


def test_research_does_not_retrieve_twice_when_the_tool_ran_but_found_nothing():
    """Finding C: `if not buffer` conflated "the agent never searched" with
    "the agent searched and found nothing", so an empty-result query ran
    retrieval twice (6 `retrieve()` calls for 3 tool calls). The guard is
    `tool.current_usage_count == 0`, which CrewAI's `BaseTool.run` increments
    before `_run` -- so it means "never invoked", nothing else.
    """
    from app.agents.tools import DocumentSearchTool

    retriever = _retriever()  # returns no chunks
    stages = _stages_with_tool_calls(retriever, calls=2)
    assert stages.research(["q1", "q2"]) == []
    assert retriever.retrieve.call_count == 2, "the fallback must not fire"


def _stages_with_tool_calls(retriever, calls):
    """CrewStages whose researcher crew invokes the real tool `calls` times."""
    from app.agents.stages import CrewStages

    stages = CrewStages(llm=MagicMock(), retriever=retriever)

    def kickoff(role, goal, description, expected_output, tools=None):
        for i in range(calls):
            tools[0].run(query=f"q{i}")
        return "searched"

    stages._kickoff = kickoff  # type: ignore[method-assign]
    return stages


def test_research_threads_top_k_into_the_direct_fallback():
    """S6: the fallback path must size retrieval the same as the tool path,
    or the chunk count would depend on whether the agent called its tool.
    """
    retriever = _retriever(_chunk("x"))
    stages, _ = _stages(retriever=retriever, outputs=["searched"])
    stages.research(["q1"], top_k=11)
    assert retriever.retrieve.call_args.kwargs["top_k"] == 11


def test_research_threads_top_k_into_the_search_tool():
    retriever = _retriever(_chunk("x"))
    stages = _stages_with_tool_calls(retriever, calls=1)
    stages.research(["q1"], top_k=11)
    assert retriever.retrieve.call_args.kwargs["top_k"] == 11


def test_grade_cap_follows_an_explicit_top_k():
    """A larger `top_k` must not leave the extra chunks silently ungraded."""
    stages, llm = _stages_for_grade(llm_outputs=["yes"] * 10)
    kept = stages.grade("q", [_chunk(f"c{i}") for i in range(10)], top_k=9)
    assert llm.call.call_count == 9
    assert len(kept) == 9


def test_research_dedupes_on_doc_id_and_text_not_text_alone():
    """Minor: identical text in two different documents is two citations."""
    retriever = MagicMock()
    same_text = "Total due: 1,452.00 EUR"
    retriever.retrieve.return_value = [
        RetrievedChunk(text=same_text, score=0.9, doc_id="doc-a", title="A", pages=[1]),
        RetrievedChunk(text=same_text, score=0.9, doc_id="doc-b", title="B", pages=[1]),
        RetrievedChunk(text=same_text, score=0.9, doc_id="doc-a", title="A", pages=[1]),
    ]
    stages, _ = _stages(retriever=retriever, outputs=["done"])
    chunks = stages.research(["q"])
    assert [c.doc_id for c in chunks] == ["doc-a", "doc-b"]


def test_research_survives_a_crew_failure():
    from app.agents.stages import CrewStages

    retriever = _retriever(_chunk("x"))
    stages = CrewStages(llm=MagicMock(), retriever=retriever)

    def boom(*args, **kwargs):
        raise RuntimeError("crew exploded")

    stages._kickoff = boom  # type: ignore[method-assign]
    assert [c.text for c in stages.research(["q1"])] == ["x"]


def _stages_for_grade(llm_outputs=None, llm_side_effect=None):
    """CrewStages whose `llm.call` (not `_kickoff`) is stubbed.

    `grade()` is the one stage that calls `self.llm.call(...)` directly
    instead of going through `_kickoff()`'s Agent/Task/Crew wrapping (see
    `grade()`'s docstring for why) -- so its tests mock the LLM
    directly rather than `_kickoff`, and record the rendered prompts via
    `call_args_list` instead of the `_kickoff`-call `calls` list the other
    stage tests use.
    """
    from app.agents.stages import CrewStages

    llm = MagicMock()
    if llm_side_effect is not None:
        llm.call.side_effect = llm_side_effect
    else:
        llm.call.side_effect = list(llm_outputs or [])
    stages = CrewStages(llm=llm, retriever=_retriever())
    return stages, llm


def test_grade_asks_once_per_chunk_and_keeps_the_yeses():
    """Ruling A/B/C (see `grade()`'s docstring): one binary verdict per
    chunk, run as a direct LLM call rather than a JSON-array call or a
    CrewAI Agent/Task -- both of those failed live. The public signature is
    unchanged -- still indices into the caller's list.
    """
    stages, llm = _stages_for_grade(llm_outputs=["no", "yes"])
    assert stages.grade("q", [_chunk("first"), _chunk("second")]) == [1]
    assert llm.call.call_count == 2, "one direct LLM call per chunk"
    prompts = [c.args[0][0]["content"] for c in llm.call.call_args_list]
    assert "first" in prompts[0] and "second" not in prompts[0]
    assert "second" in prompts[1]


def test_grade_keeps_a_mixed_batch_only_relevant():
    """A relevant chunk and a clearly irrelevant one in the same batch:
    only the relevant index comes back.
    """
    stages, _ = _stages_for_grade(llm_outputs=["YES", "NO"])
    kept = stages.grade("q", [_chunk("relevant one"), _chunk("irrelevant one")])
    assert kept == [0]


def test_grade_keeps_unparseable_verdicts_failing_open():
    stages, _ = _stages_for_grade(llm_outputs=["I am not sure about this one", "no"])
    assert stages.grade("q", [_chunk("a"), _chunk("b")]) == [0]


def test_grade_keeps_a_chunk_whose_llm_call_raised():
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("llm call failed")
        return "no"

    stages, _ = _stages_for_grade(llm_side_effect=flaky)
    # Chunk 0's grading blew up -> kept rather than silently dropped.
    assert stages.grade("q", [_chunk("a"), _chunk("b")]) == [0]


def test_grade_respects_a_unanimous_no():
    stages, _ = _stages_for_grade(llm_outputs=["no", "no"])
    assert stages.grade("q", [_chunk("a"), _chunk("b")]) == []


def test_grade_caps_the_number_of_chunks_and_prefers_high_scores():
    """Ruling A: bounded on CPU -- at most retrieval_top_k chunks, highest
    scoring first, and the returned indices still address the caller's list.
    """
    stages, llm = _stages_for_grade(llm_outputs=["yes"] * 10)
    chunks = [_chunk(f"c{i}") for i in range(10)]
    chunks[7].score = 0.99  # highest
    chunks[3].score = 0.98
    kept = stages.grade("q", chunks)
    assert llm.call.call_count == 6, "retrieval_top_k defaults to 6"
    assert 7 in kept and 3 in kept
    assert kept == sorted(kept) and all(0 <= i < 10 for i in kept)


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
