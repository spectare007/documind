"""Phoenix OTLP tracing setup.

Registers an OpenTelemetry TracerProvider that exports spans to Arize
Phoenix over OTLP-HTTP under the `documind` project, then attempts to attach
OpenInference/OpenTelemetry auto-instrumentation for FastAPI (the ASGI
request layer), CrewAI, LlamaIndex, LiteLLM and the OpenAI SDK, so both
inbound HTTP requests and agent/LLM calls (Tasks 9-11) show up as spans
automatically.

WHY BOTH LiteLLM *AND* OpenAI INSTRUMENTORS (fix for a review finding):
"trace all inference calls" was not actually satisfied before. On crewai
1.x, `crewai.LLM(model="ollama/...", base_url=...)` is a factory that
returns a *native provider* -- `OpenAICompatibleCompletion` -- which calls
the `openai` SDK directly against Ollama's OpenAI-compatible `/v1`
endpoint. It never touches LiteLLM; litellm is not even installed (it
became an optional `crewai[litellm]` extra), so `LiteLLMInstrumentor`
attaches to nothing and logs a `DependencyConflict` at startup. The result
was a Phoenix trace tree with CHAIN/AGENT/TOOL spans but *zero* LLM spans:
no prompts, no completions, no token counts. `OpenAIInstrumentor` patches
the SDK crewai actually calls, which restores them. LiteLLM's instrumentor
is kept because it costs nothing when absent and would cover any future
model that does route through LiteLLM.

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
tests). Each instrumentor is attempted independently: a CrewAI-version
mismatch (or any other single instrumentor failure) must not prevent the
other instrumentors from being attached.

TWO DIFFERENT LIFETIMES, TWO DIFFERENT GUARDS (fix for a review finding):
`register()` and the process-wide OpenInference instrumentors (CrewAI,
LlamaIndex, LiteLLM) really are process-global and correctly run at most
once -- re-registering a tracer provider or re-patching a third-party
*library* on every `create_app()` call would be wasteful and, for some
instrumentors, unsafe. Those stay behind the module-level `_initialized`
flag, exactly as before. FastAPI/ASGI instrumentation is different: it
patches a *specific app instance's* `build_middleware_stack`, so gating it
behind the same process-wide flag means every `create_app()` after the
first produces a completely untraced app -- no root server span, ever, for
the lifetime of the process. This bit in practice: `app.main`'s
module-level `app = create_app()` (needed for `uvicorn app.main:app`)
consumes the one-shot flag on import, so any *second* `create_app()` in the
same process -- e.g. every `TestClient(create_app())` fixture in this test
suite -- silently got a `NonRecordingSpan` (trace_id all zeros) for every
request, while direct callers of the module-level `app` got a real span.
`setup_tracing(app)` therefore now instruments the given `app` on *every*
call, independent of `_initialized`. This is safe to repeat because
`FastAPIInstrumentor.instrument_app()` carries its own per-app guard
(`app._is_instrumented_by_opentelemetry`) -- instrumenting the same app
twice is already a no-op by the library's own design, so the extra
process-wide flag around this specific step bought nothing and broke every
second app instance. See `tests/unit/test_tracing.py::
test_second_create_app_in_same_process_still_gets_a_real_traced_span` for
the regression test (in-memory exporter, asserts on the exported span's
actual trace id, not just that instrumentation "succeeded" without an
exception).

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
# Cached by the one-shot process-wide setup below so every later per-app
# FastAPI instrumentation call (see `setup_tracing`'s per-`app` step) uses
# the same registered provider instead of falling back to the OTel global
# default. Stays `None` if `register()` never succeeded (e.g. Phoenix
# package incompatible) -- FastAPI instrumentation is still attempted in
# that case, just without an explicit provider.
_tracer_provider = None


def setup_tracing(app: "FastAPI | None" = None) -> None:
    """Set up Phoenix OTLP tracing and auto-instrumentation.

    Two independent steps, two independent guards -- see the module
    docstring's "TWO DIFFERENT LIFETIMES" section for the full rationale:

    1. Process-wide setup (tracer-provider `register()` plus the CrewAI/
       LlamaIndex/LiteLLM OpenInference instrumentors): runs at most once
       per process, guarded by the module-level `_initialized` flag.
    2. Per-app FastAPI/ASGI instrumentation: runs on *every* call that
       passes an `app`, regardless of `_initialized`, because it patches
       that specific app instance's `build_middleware_stack`
       (`app.main.create_app()` passes its own `app` -- see the module
       docstring's "CALLER TIMING REQUIREMENT" for why this must happen
       before `app` is returned, not from inside `lifespan()`). Safe to
       repeat: `FastAPIInstrumentor.instrument_app()` no-ops on an app it
       has already instrumented.

    Omitting `app` (e.g. in tests that only care about the OpenInference
    instrumentors) skips step 2 but still attempts step 1. Both steps are
    best-effort and never raise.
    """
    global _initialized, _tracer_provider
    if not _initialized:
        try:
            from phoenix.otel import register

            _tracer_provider = register(
                project_name="documind",
                endpoint=f"{get_settings().phoenix_base_url}/v1/traces",
                batch=True,
                set_global_tracer_provider=True,
                verbose=False,
            )
        except Exception as exc:
            # Deliberately does NOT set `_initialized = True` here (matches
            # pre-existing behavior): if Phoenix/the SDK is unreachable or
            # incompatible now but recovers later, the next `setup_tracing()`
            # call in this process retries process-wide setup from scratch.
            logger.warning("phoenix tracing setup skipped: %s", exc)
            _tracer_provider = None
        else:
            env_before = set(os.environ)
            succeeded, failed = [], []
            for name, instrument in _process_wide_instrumentors():
                try:
                    instrument(_tracer_provider)
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

    if app is not None:
        _instrument_fastapi_app(app, _tracer_provider)


def _instrument_fastapi_app(app: "FastAPI", tracer_provider) -> None:
    """Instrument one FastAPI app instance for request-level server spans.

    Deliberately outside the process-wide `_initialized` guard (see
    `setup_tracing`'s docstring) -- called on every `setup_tracing(app)`
    invocation. `FastAPIInstrumentor.instrument_app` carries its own
    per-app `_is_instrumented_by_opentelemetry` guard, so calling this
    again for an app already instrumented is already a safe no-op; the
    outer try/except here only guards against a genuinely failed/absent
    instrumentor package, consistent with "tracing must never raise".
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    except Exception as exc:
        logger.warning("fastapi instrumentation failed for this app: %s", exc)


def _process_wide_instrumentors():
    """Yield (name, instrument_fn) pairs for the one-shot, process-wide
    OpenInference instrumentors (everything except FastAPI/ASGI, which is
    per-app -- see `_instrument_fastapi_app`).

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

    def _openai(tracer_provider):
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    return [
        ("crewai", _crewai),
        ("llama_index", _llama_index),
        ("litellm", _litellm),
        ("openai", _openai),
    ]
