from pathlib import Path
from unittest.mock import MagicMock

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def _manager(**kw):
    from app.observability.prompts import PromptManager
    return PromptManager(prompts_dir=PROMPTS_DIR, **kw)


def test_loads_and_formats_yaml():
    m = _manager()
    out = m.get("router", question="hello there")
    assert "hello there" in out and "{question}" not in out


def test_all_five_prompts_load():
    m = _manager()
    for name in ["router", "rewriter", "grader", "synthesizer", "hallucination_checker"]:
        assert m.template(name)


def test_phoenix_override_wins():
    m = _manager()
    m._phoenix_templates["router"] = "OVERRIDE {question}"
    assert m.get("router", question="x") == "OVERRIDE x"


def test_phoenix_failure_falls_back_to_yaml():
    m = _manager()
    m._client = MagicMock()
    m._client.prompts.get.side_effect = RuntimeError("phoenix down")
    m.refresh_from_phoenix()          # must not raise
    assert "rag" in m.get("router", question="q")
