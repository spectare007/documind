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
