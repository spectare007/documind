"""Phoenix OTLP tracing setup.

Registers an OpenTelemetry TracerProvider that exports spans to Arize
Phoenix over OTLP-HTTP under the `documind` project, then attempts to attach
OpenInference/OpenTelemetry auto-instrumentation for FastAPI (the ASGI
request layer), CrewAI, LlamaIndex, and LiteLLM, so both inbound HTTP
requests and agent/LLM calls (Tasks 9-11) show up as spans automatically.

The FastAPI instrumentor is what actually creates the root "server span" for
each HTTP request -- without it there is no active OTel span during request
handling at all, so `CorrelationIdMiddleware`'s `trace.get_current_span()`
call would always see a no-op, non-recording span and the `correlation.id`
attribute would never actually land anywhere. `FastAPIInstrumentor.
instrument_app(app, ...)` wraps the *specific* app instance's ASGI callable
outside of every `add_middleware`-registered middleware by monkey-patching
`app.build_middleware_stack` so `OpenTelemetryMiddleware` sits outside
`ServerErrorMiddleware -> [user middlewares] -> router`, so the server span
is already current by the time `CorrelationIdMiddleware.dispatch()` runs,
both before and after `call_next()` -- confirmed with an `InMemorySpanExporter`
in `tests/unit/test_middleware.py`.

CALLER TIMING REQUIREMENT: `setup_tracing(app)` must be called before `app`
receives its first ASGI call of *any* kind. Starlette's `Starlette.__call__`
does `if self.middleware_stack is None: self.middleware_stack =
self.build_middleware_stack()` -- and for uvicorn, the very first such call
is the ASGI *lifespan* startup call, i.e. exactly the call that runs our
`lifespan()` context manager. Calling `setup_tracing(app)` from inside
`lifespan()` therefore patches `build_middleware_stack` **after** Starlette
has already built and cached the original, uninstrumented stack on that same
call -- `_is_instrumented_by_opentelemetry` gets set and no exception is
raised, but the patched method is never invoked again, so real HTTP requests
silently get no server span at all. `app.main.create_app()` calls
`setup_tracing(app)` synchronously, before returning `app`, specifically to
avoid this trap.

Tracing is always best-effort: `setup_tracing()` must never raise, must
return quickly even when Phoenix is unreachable (span export is async/
batched, so `register()` does not block on connectivity), and must be safe
to call more than once (e.g. once per FastAPI app instance created in
tests) -- a module-level flag makes it a no-op after the first successful
call. Each instrumentor is attempted independently: a CrewAI-version
mismatch (or any other single instrumentor failure) must not prevent the
other instrumentors from being attached.

--- Deliberate, documented exception to "no os.environ reads in app code" ---

Importing CrewAI (transitively, via its OpenInference instrumentor) runs
`crewai.llm`'s module-level `dotenv.load_dotenv()`, which walks up from the
current working directory and -- in this repo layout -- finds and loads the
repo-root `.env` (meant for docker-compose variable substitution) into the
real process environment, e.g. leaking `DOCUMIND_API_KEY=change-me`. Since
real env vars outrank pydantic-settings' own `env_file=".env"` lookup, that
silently overrides every `get_settings()` call for the rest of the process
(this was caught because it broke API-key auth in tests that ran after
tracing setup). `setup_tracing()` snapshots `os.environ` before
instrumenting and scrubs (`_scrub_env_pollution`, from `app.core.env_guard`)
any keys that appear afterward, so this specific third-party import side
effect can never leak into our config.

`app.core.env_guard` is the single place in the codebase that reads/mutates
`os.environ`, holds the full rationale, and is shared with `app.agents`,
which imports CrewAI itself (not just its instrumentor) and needs exactly
the same containment.
"""

import logging
import os
from typing import TYPE_CHECKING

from app.core.config import get_settings

# Re-exported under its historical private name: this module's callers and
# tests refer to `_scrub_env_pollution`, while the implementation (and the
# full rationale) now lives in `app.core.env_guard` so `app.agents` -- which
# imports CrewAI directly, not just its instrumentor -- can share it.
from app.core.env_guard import scrub_env_pollution as _scrub_env_pollution

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)
_initialized = False


def setup_tracing(app: "FastAPI | None" = None) -> None:
    """Set up Phoenix OTLP tracing and auto-instrumentation.

    `app`, when given, is the actual FastAPI instance to instrument for
    request-level spans (`app.main.create_app()` passes its own `app`
    argument -- see the module docstring's "CALLER TIMING REQUIREMENT" for
    why this must happen before `app` is returned, not from inside
    `lifespan()`). Omitting it (e.g. in tests that only care about the
    OpenInference instrumentors) skips FastAPI/ASGI instrumentation but
    still attempts the others -- still idempotent and non-raising either
    way.
    """
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
    for name, instrument in _instrumentors(app):
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


def _instrumentors(app: "FastAPI | None"):
    """Yield (name, instrument_fn) pairs, each importing its own instrumentor.

    Imports happen lazily and per-instrumentor so that one package being
    absent or incompatible with the installed FastAPI/CrewAI/LlamaIndex
    version can't prevent the others from being attempted.
    """

    def _fastapi(tracer_provider):
        if app is None:
            raise RuntimeError("setup_tracing() called without an app; skipping FastAPI instrumentation")
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

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
        ("fastapi", _fastapi),
        ("crewai", _crewai),
        ("llama_index", _llama_index),
        ("litellm", _litellm),
    ]
