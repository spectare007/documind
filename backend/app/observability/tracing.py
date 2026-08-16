"""Phoenix OTLP tracing setup.

Registers an OpenTelemetry TracerProvider that exports spans to Arize
Phoenix over OTLP-HTTP under the `documind` project, then attempts to attach
OpenInference auto-instrumentation for CrewAI, LlamaIndex, and LiteLLM so
agent/LLM calls made in later tasks show up as spans automatically.

Tracing is always best-effort: `setup_tracing()` must never raise, must
return quickly even when Phoenix is unreachable (span export is async/
batched, so `register()` does not block on connectivity), and must be safe
to call more than once (e.g. once per FastAPI app instance created in
tests) -- a module-level flag makes it a no-op after the first successful
call. Each OpenInference instrumentor is attempted independently: a
CrewAI-version mismatch (or any other single instrumentor failure) must not
prevent the other instrumentors from being attached.

Importing CrewAI (transitively, via its OpenInference instrumentor) runs
`crewai.llm`'s module-level `dotenv.load_dotenv()`, which walks up from the
current working directory and -- in this repo layout -- finds and loads the
repo-root `.env` (meant for docker-compose variable substitution) into the
real process environment, e.g. leaking `DOCUMIND_API_KEY`. Since real env
vars outrank pydantic-settings' own `env_file=".env"` lookup, that would
silently override every `get_settings()` call for the rest of the process
(observed breaking API-key auth in tests that run after tracing setup).
`setup_tracing()` snapshots `os.environ` before instrumenting and scrubs any
keys that appear afterward, so third-party import side effects can never
leak into our config -- while env vars a real deployment set before startup
(already present at snapshot time) are left untouched.
"""

import logging
import os

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_initialized = False


def setup_tracing() -> None:
    global _initialized
    if _initialized:
        return
    try:
        from phoenix.otel import register

        tracer_provider = register(
            project_name="documind",
            endpoint=f"{get_settings().phoenix_base_url}/v1/traces",
            batch=True,
            set_global_tracer_provider=True,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("phoenix tracing setup skipped: %s", exc)
        return

    env_before = set(os.environ)
    succeeded, failed = [], []
    for name, instrument in _instrumentors():
        try:
            instrument(tracer_provider)
            succeeded.append(name)
        except Exception as exc:
            failed.append(name)
            logger.warning("openinference instrumentor %s failed: %s", name, exc)
    _scrub_env_pollution(env_before)

    _initialized = True
    logger.info(
        "phoenix tracing initialized (project=documind); instrumentors ok=%s failed=%s",
        succeeded,
        failed,
    )


def _scrub_env_pollution(env_before: set[str]) -> None:
    """Remove env vars that appeared during instrumentation imports.

    See module docstring: CrewAI's own `load_dotenv()` can inject vars from
    an unrelated `.env` file found by walking up from cwd. Only vars that
    were absent before instrumentation and present after are removed --
    anything the deployment genuinely set before startup is left alone.
    """
    leaked = set(os.environ) - env_before
    for key in leaked:
        del os.environ[key]
    if leaked:
        logger.warning(
            "scrubbed %d env var(s) leaked by instrumentation imports: %s",
            len(leaked),
            sorted(leaked),
        )


def _instrumentors():
    """Yield (name, instrument_fn) pairs, each importing its own instrumentor.

    Imports happen lazily and per-instrumentor so that one package being
    absent or incompatible with the installed CrewAI/LlamaIndex/LiteLLM
    version can't prevent the others from being attempted.
    """

    def _crewai(tracer_provider):
        from openinference.instrumentation.crewai import CrewAIInstrumentor

        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

    def _llama_index(tracer_provider):
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    def _litellm(tracer_provider):
        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    return [
        ("crewai", _crewai),
        ("llama_index", _llama_index),
        ("litellm", _litellm),
    ]
