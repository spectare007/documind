from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.pipelines.types import RetrievedChunk

# Repo-root prompts/ dir: in production the container's cwd is /app with
# prompts/ mounted alongside it (docker-compose.yml), so Settings.prompts_dir
# defaults to a bare relative "prompts". Under pytest the cwd is backend/, so
# SimplePipeline's PromptManager needs to be pointed at the real directory --
# same convention as tests/unit/test_api_documents.py.
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@pytest.fixture(autouse=True)
def _prompts_dir(monkeypatch):
    monkeypatch.setenv("DOCUMIND_PROMPTS_DIR", str(PROMPTS_DIR))


def test_simple_pipeline_answers_with_citations():
    from app.pipelines.simple import SimplePipeline
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(text="[Doc > A]\n\nTotal: 100", score=0.9,
                       doc_id="d", title="Doc", section_path="A", pages=[1]),
    ]
    llm = MagicMock()
    llm.complete.return_value = MagicMock(text="The total is 100 [Doc, A].")
    statuses = []
    result = SimplePipeline(retriever=retriever, llm=llm).answer(
        "what is the total?", history=[], on_status=statuses.append
    )
    assert result.answer.startswith("The total is 100")
    assert result.citations[0].title == "Doc"
    assert result.grounded is None and result.retrieval_attempts == 1
    assert any("Retrieving" in s for s in statuses)
    prompt_sent = llm.complete.call_args.args[0]
    assert "Total: 100" in prompt_sent and "what is the total?" in prompt_sent


def test_simple_pipeline_no_chunks_message():
    from app.pipelines.simple import SimplePipeline
    retriever = MagicMock(); retriever.retrieve.return_value = []
    llm = MagicMock()
    result = SimplePipeline(retriever=retriever, llm=llm).answer("q", history=[])
    assert "couldn't find" in result.answer.lower()
    llm.complete.assert_not_called()
