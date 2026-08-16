"""Pure-function tests for the agentic pipeline's LLM output parsers.

These lock in the *safe defaults* that keep the corrective-RAG loop from
degrading into silence when a small local model answers off-format: routing
defaults to retrieval, grading defaults to keeping everything, and the
groundedness check fails open.
"""

from app.agents.stages import (
    parse_indices,
    parse_queries,
    parse_relevance_verdict,
    parse_route,
    parse_verdict,
)


def test_parse_route():
    assert parse_route("rag") == "rag"
    assert parse_route(" Direct.\n") == "direct"
    assert parse_route("gibberish") == "rag"  # safe default: retrieve
    # Anchored, not a substring search: a sentence that merely mentions the
    # word must not flip the route away from retrieval.
    assert parse_route("This is not a direct question") == "rag"
    assert parse_route("The answer is indirect") == "rag"
    assert parse_route("DIRECT") == "direct"


def test_parse_queries():
    assert parse_queries('["a", "b"]') == ["a", "b"]
    assert parse_queries('Here you go: ["invoice total June"]') == ["invoice total June"]
    assert parse_queries("not json", fallback="orig q") == ["orig q"]


def test_parse_indices():
    assert parse_indices("[0, 2]", n_chunks=3) == [0, 2]
    assert parse_indices("[0, 9]", n_chunks=3) == [0]   # out of range dropped
    assert parse_indices("none relevant []", n_chunks=3) == []
    assert parse_indices("garbage", n_chunks=2) == [0, 1]  # unparseable: keep all
    # Observed live: given a single chunk, qwen2.5:3b copies the "e.g. [0, 2]"
    # example straight out of the grader prompt. Every index out of range is a
    # malformed reply, not a judgment of irrelevance, so it fails open too --
    # otherwise the pipeline answers "I couldn't find anything" for a corpus
    # that does contain the answer.
    assert parse_indices("[2]", n_chunks=1) == [0]
    assert parse_indices("[7, 9]", n_chunks=2) == [0, 1]


def test_parse_verdict():
    assert parse_verdict("yes") is True
    assert parse_verdict(" No\n") is False
    assert parse_verdict("unclear") is True  # fail open: do not block answers


def test_parse_relevance_verdict():
    assert parse_relevance_verdict("YES") is True
    assert parse_relevance_verdict(" no\n") is False
    assert parse_relevance_verdict("yes, this helps") is True
    assert parse_relevance_verdict("No, unrelated") is False
    assert parse_relevance_verdict("unclear") is True  # fail open: keep the chunk


def test_parse_relevance_verdict_does_not_repeat_the_nested_substring_bug():
    """Regression test for a real near miss in the grader's history: an earlier grader asked for RELEVANT/IRRELEVANT and
    reused `parse_verdict`'s "no"-prefix check against that vocabulary.
    "IRRELEVANT" does not start with "no", so every chunk was silently kept
    regardless of the model's actual verdict -- a rubber-stamp grader
    dressed up as a discriminating one. "relevant" is a literal substring of
    "irrelevant", which is exactly the trap a naive `"relevant" in text`
    (or a check written for the wrong vocabulary) falls into.

    This parser is written for its own yes/no vocabulary and is anchored
    with `startswith`, not a substring search, so a reply that happens to
    contain "relevant"/"irrelevant" text (rather than yes/no) is simply
    unparseable and fails open -- it is never mistaken for an affirmative
    "yes" by matching a nested substring.
    """
    assert parse_relevance_verdict("IRRELEVANT") is True  # unparseable -> fails open
    assert parse_relevance_verdict("RELEVANT") is True  # unparseable -> fails open, not
    # "matched because it starts with yes" -- it doesn't start with "yes" either.
    assert not "RELEVANT".lower().startswith("yes")
