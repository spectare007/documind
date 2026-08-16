from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


class _FakePhoenixPrompts:
    """Minimal stand-in for `client.prompts` that behaves like the real
    Phoenix prompt store closely enough to test sync idempotency: `get`
    raises ValueError (mirroring the real client's 404 behavior) until a
    prompt has been `create`-d, then returns an object whose `.format()`
    reports back the content it was created with.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.create_calls = 0

    def get(self, *, prompt_identifier: str):
        if prompt_identifier not in self.store:
            raise ValueError(f"Prompt not found: {prompt_identifier}")
        content = self.store[prompt_identifier]
        version = MagicMock()
        version.format.return_value = {"messages": [{"role": "system", "content": content}]}
        return version

    def create(self, *, name: str, version, prompt_description=None):
        self.create_calls += 1
        content = version.format()["messages"][0]["content"]
        self.store[name] = content
        return MagicMock()


class _FakePhoenixClient:
    def __init__(self) -> None:
        self.prompts = _FakePhoenixPrompts()


def test_sync_to_phoenix_is_idempotent_on_content():
    m = _manager()
    fake_client = _FakePhoenixClient()
    m._client = fake_client

    m.sync_to_phoenix()
    assert fake_client.prompts.create_calls == 5  # nothing existed yet -> 5 creates

    m.sync_to_phoenix()
    assert fake_client.prompts.create_calls == 5  # unchanged content -> no new versions


def test_sync_to_phoenix_creates_new_version_when_content_changes():
    m = _manager()
    fake_client = _FakePhoenixClient()
    m._client = fake_client

    m.sync_to_phoenix()
    assert fake_client.prompts.create_calls == 5

    m._yaml["router"]["template"] = "A completely different router template {question}"
    m.sync_to_phoenix()
    assert fake_client.prompts.create_calls == 6  # only the changed prompt got a new version


def test_get_raises_on_unresolved_placeholder():
    m = _manager()
    with pytest.raises(ValueError, match="question"):
        m.get("router")  # router.yaml declares {question}; none supplied


def test_get_tolerates_braces_inside_substituted_values():
    m = _manager()
    out = m.get("router", question="what is {context} anyway?")
    assert "what is {context} anyway?" in out
