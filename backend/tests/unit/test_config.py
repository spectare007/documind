from pathlib import Path


def test_defaults():
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.llm_model == "qwen2.5:3b"
    assert s.embed_model == "nomic-embed-text"
    assert s.embed_dim == 768
    # `simple` on measured behaviour: agentic mode answered 15 of 23
    # answerable golden-set questions at a median of 125s, simple mode
    # answered every question it was tried against, in 25 to 82s, from the
    # same index (doc/evaluation-report.md).
    assert s.pipeline_mode == "simple"
    assert s.max_retrieval_attempts == 2
    assert s.request_budget_seconds == 300.0
    assert s.data_dir == Path("data/documents")


def test_env_override(monkeypatch):
    monkeypatch.setenv("DOCUMIND_LLM_MODEL", "llama3.2:3b")
    monkeypatch.setenv("DOCUMIND_PIPELINE_MODE", "agentic")
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.llm_model == "llama3.2:3b"
    assert s.pipeline_mode == "agentic", "agentic stays fully supported, opt-in"


def test_get_settings_cached():
    from app.core.config import get_settings
    assert get_settings() is get_settings()
