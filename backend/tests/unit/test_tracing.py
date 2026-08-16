"""Tests for `app.observability.tracing`'s env-var scrub.

`setup_tracing()` snapshots `os.environ` before running the instrumentor
loop and deletes any key that appears afterward -- a defensive guard against
CrewAI's own `load_dotenv()` (triggered by importing it) picking up the
repo-root `.env` and leaking config like `DOCUMIND_API_KEY` into the real
process environment. These tests pin that behaviour directly against
`_scrub_env_pollution`, independent of network/Phoenix availability.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_test_env_vars():
    """Belt-and-suspenders cleanup in case a test fails mid-mutation."""
    yield
    os.environ.pop("DOCUMIND_TEST_PRE_EXISTING", None)
    os.environ.pop("DOCUMIND_TEST_LEAKED_DURING_INSTRUMENTATION", None)


def test_scrub_removes_only_newly_appeared_keys(monkeypatch):
    from app.observability.tracing import _scrub_env_pollution

    monkeypatch.setenv("DOCUMIND_TEST_PRE_EXISTING", "keep-me")
    env_before = set(os.environ)

    # Simulate a third-party import (e.g. crewai.llm's load_dotenv()) that
    # injects a var that was not present at snapshot time.
    os.environ["DOCUMIND_TEST_LEAKED_DURING_INSTRUMENTATION"] = "change-me"

    _scrub_env_pollution(env_before)

    assert os.environ["DOCUMIND_TEST_PRE_EXISTING"] == "keep-me"
    assert "DOCUMIND_TEST_LEAKED_DURING_INSTRUMENTATION" not in os.environ


def test_scrub_is_a_noop_when_nothing_leaked(monkeypatch):
    from app.observability.tracing import _scrub_env_pollution

    monkeypatch.setenv("DOCUMIND_TEST_PRE_EXISTING", "keep-me")
    env_before = set(os.environ)

    _scrub_env_pollution(env_before)  # nothing new appeared

    assert os.environ["DOCUMIND_TEST_PRE_EXISTING"] == "keep-me"


def test_openai_instrumentor_is_registered_and_importable():
    """Ruling B: "trace all inference calls" was not satisfied before.

    crewai 1.x routes `ollama/...` to its native `OpenAICompatibleCompletion`,
    which calls the `openai` SDK directly and never touches LiteLLM (which
    isn't even installed), so Phoenix showed agent/tool spans but zero LLM
    spans. `OpenAIInstrumentor` patches the SDK that is actually called.
    Each instrumentor must stay independently importable so one failure
    cannot disable the others.
    """
    from app.observability.tracing import _process_wide_instrumentors

    registered = dict(_process_wide_instrumentors())
    assert "openai" in registered
    # The instrument fn imports lazily; check the package is actually present
    # rather than silently no-oping the whole point of this fix.
    from openinference.instrumentation.openai import OpenAIInstrumentor

    assert OpenAIInstrumentor is not None


def test_one_failing_instrumentor_does_not_disable_the_others(monkeypatch):
    """Each entry is attempted in its own try/except."""
    import app.observability.tracing as tracing_module

    attempted = []

    def boom(_tp):
        attempted.append("crewai")
        raise RuntimeError("version mismatch")

    def ok(_tp):
        attempted.append("openai")

    monkeypatch.setattr(tracing_module, "_process_wide_instrumentors",
                        lambda: [("crewai", boom), ("openai", ok)])
    monkeypatch.setattr(tracing_module, "_initialized", False)
    monkeypatch.setattr(tracing_module, "_tracer_provider", None)
    monkeypatch.setattr("phoenix.otel.register", lambda **kw: None)

    tracing_module.setup_tracing()  # must not raise

    assert attempted == ["crewai", "openai"]


def test_no_env_pollution_context_manager_cleans_up_even_on_error(monkeypatch):
    """`app.agents` wraps its `import crewai` in this; a failed import must
    still not leave the process environment poisoned.
    """
    from app.core.env_guard import no_env_pollution

    monkeypatch.setenv("DOCUMIND_TEST_PRE_EXISTING", "keep-me")

    with pytest.raises(RuntimeError), no_env_pollution():
        os.environ["DOCUMIND_TEST_LEAKED_DURING_INSTRUMENTATION"] = "change-me"
        raise RuntimeError("import blew up halfway")

    assert "DOCUMIND_TEST_LEAKED_DURING_INSTRUMENTATION" not in os.environ
    assert os.environ["DOCUMIND_TEST_PRE_EXISTING"] == "keep-me"


def test_second_create_app_in_same_process_still_gets_a_real_traced_span(monkeypatch):
    """Regression test for the review finding: gating per-app FastAPI
    instrumentation behind the process-wide `_initialized` flag meant every
    *second* `setup_tracing(app)` call in a process (e.g. every
    `TestClient(create_app())` fixture, once `app.main`'s module-level
    `app = create_app()` has already consumed the flag on import) produced
    a completely untraced app: no root server span, so `/query`'s trace_id
    came back all zeros under tests even though it was real under uvicorn.

    Proves the fix at the same rigor as
    `test_middleware.py::test_correlation_id_lands_on_a_real_recording_span`:
    a real `InMemorySpanExporter`, asserting on the actual exported span's
    trace id -- not just that `setup_tracing()` ran without raising.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import app.observability.tracing as tracing

    # Simulate process-wide setup having already completed for a *different*
    # app -- e.g. `app.main`'s module-level `app = create_app()` on import --
    # which is exactly the state that made the second app go untraced.
    # `monkeypatch` restores both module globals automatically on teardown,
    # so this can't leak into other tests' tracer-provider state.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_tracer_provider", provider)

    first_app = FastAPI()
    tracing.setup_tracing(first_app)

    second_app = FastAPI()

    @second_app.get("/ping")
    def ping():
        return {"ok": True}

    tracing.setup_tracing(second_app)  # the call under test

    try:
        r = TestClient(second_app).get("/ping")
        assert r.status_code == 200
    finally:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.uninstrument_app(first_app)
        FastAPIInstrumentor.uninstrument_app(second_app)

    server_spans = [s for s in exporter.get_finished_spans() if s.name == "GET /ping"]
    assert len(server_spans) == 1
    trace_id = server_spans[0].context.trace_id
    assert trace_id != 0
    assert format(trace_id, "032x") != "0" * 32


def test_importing_the_agents_package_does_not_pollute_the_environment():
    """Regression guard for the real failure this caused: `import crewai`
    runs `load_dotenv()`, which loaded the repo-root `.env` and overrode
    `DOCUMIND_API_KEY` process-wide, breaking API-key auth for every test
    that ran afterwards. Run in a subprocess so the check is unaffected by
    whatever this session already imported.
    """
    import subprocess
    import sys

    probe = (
        "import os; before = set(os.environ);"
        " import app.agents;"
        " print(sorted(set(os.environ) - before))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout
