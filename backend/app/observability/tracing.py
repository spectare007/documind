"""Phoenix OTLP tracing setup.

Registers an OpenTelemetry TracerProvider that exports spans to Arize
Phoenix over OTLP-HTTP under the `documind` project, then attempts to attach
OpenInference/OpenTelemetry auto-instrumentation for FastAPI (the ASGI
request layer), CrewAI, LlamaIndex, LiteLLM and the OpenAI SDK, so both
inbound HTTP requests and agent/LLM calls show up as spans automatically.

Both LiteLLM and OpenAI instrumentors are attached: on crewai 1.x,
`crewai.LLM(model="ollama/...", ...)` calls the `openai` SDK directly
against Ollama's OpenAI-compatible endpoint rather than going through
LiteLLM (which isn't even installed by default), so `OpenAIInstrumentor` is
what actually produces LLM spans (prompts, completions, token counts).
`LiteLLMInstrumentor` is kept in case a future model routes through it; it
costs nothing when its target library is absent.

The FastAPI instrumentor creates the root "server span" for each HTTP
request -- without it `CorrelationIdMiddleware`'s `trace.get_current_span()`
always sees a no-op span and `correlation.id` never lands anywhere.
`FastAPIInstrumentor.instrument_app()` places `OpenTelemetryMiddleware`
outside every `add_middleware`-registered middleware by monkey-patching
`app.build_middleware_stack`, confirmed with an `InMemorySpanExporter` in
`tests/unit/test_middleware.py`.

CALLER TIMING REQUIREMENT: `setup_tracing(app)` must run before `app`
receives its first ASGI call of *any* kind. Starlette caches
`self.middleware_stack` on that very first call -- which, for uvicorn, is
the ASGI *lifespan* startup call, i.e. the same call that runs our
`lifespan()` context manager. Instrumenting from inside `lifespan()`
therefore patches `build_middleware_stack` **after** Starlette has already
built and cached the uninstrumented stack: `_is_instrumented_by_opentelemetry`
gets set, no exception is raised, but the patched method is never invoked
again, so real requests silently get no server span at all.
`app.main.create_app()` calls `setup_tracing(app)` synchronously, before
returning `app`, specifically to avoid this trap.

Tracing is always best-effort: `setup_tracing()` must never raise, must
return quickly even when Phoenix is unreachable, and must be safe to call
more than once (e.g. once per FastAPI app instance created in tests). Each
instrumentor is attempted independently: a CrewAI-version mismatch (or any
other single instrumentor failure) must not prevent the rest.

TWO DIFFERENT LIFETIMES, TWO DIFFERENT GUARDS: `register()` and the
process-wide OpenInference instrumentors (CrewAI, LlamaIndex, LiteLLM) are
genuinely process-global, so they run at most once, behind the module-level
`_initialized` flag. FastAPI/ASGI instrumentation is different: it patches a
*specific app instance's* `build_middleware_stack`, so gating it behind the
same flag left every `create_app()` after the first completely untraced --
in practice, every `TestClient(create_app())` fixture in this suite got a
`NonRecordingSpan` (trace_id all zeros) while the module-level `app` (which
consumed the one-shot flag on import) got a real one. `setup_tracing(app)`
therefore instruments the given `app` on *every* call, independent of
`_initialized`; this is safe because `FastAPIInstrumentor.instrument_app()`
already no-ops on an app it has instrumented before. See
`tests/unit/test_tracing.py::
test_second_create_app_in_same_process_still_gets_a_real_traced_span`.

--- Deliberate, documented exception to "no os.environ reads in app code" ---

Importing CrewAI (transitively, via its OpenInference instrumentor) runs
`crewai.llm`'s module-level `dotenv.load_dotenv()`, which walks up from the
current working directory and -- in this repo layout -- loads the repo-root
`.env` (meant only for docker-compose variable substitution) into the real
process environment, e.g. leaking `DOCUMIND_API_KEY=change-me`. Real env
vars outrank pydantic-settings' own `env_file` lookup, so this silently
overrode every `get_settings()` call for the rest of the process (caught
because it broke API-key auth in tests run after tracing setup).
`setup_tracing()` snapshots `os.environ` before instrumenting and scrubs
(`_scrub_env_pollution`, from `app.core.env_guard`) any keys that appear
afterward, so this side effect can never leak into our config.

`app.core.env_guard` is the single place in the codebase that reads/mutates
`os.environ`, holds the full rationale, and is shared with `app.agents`,
which imports CrewAI directly and needs the same containment.
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


def _trace_config():
    """Build the OpenInference `TraceConfig` that honors `Settings.trace_content`.

    Uses OpenInference's own masking mechanism rather than a bespoke
    redaction layer: when `trace_content` is False, `hide_inputs`/
    `hide_outputs` redact LLM prompts, completions and retrieved chunk text,
    and `hide_embeddings_text` redacts the raw text sent to the embedding
    model. Span names, timings, status and token counts are on separate
    attributes `TraceConfig` never masks, so those still export either way.
    Called fresh by each instrumentor closure below (not cached) so a
    `Settings` change under test is picked up without a restart.
    """
    from openinference.instrumentation import TraceConfig

    capture = get_settings().trace_content
    return TraceConfig(
        hide_inputs=not capture,
        hide_outputs=not capture,
        hide_embeddings_text=not capture,
    )


def _process_wide_instrumentors():
    """Yield (name, instrument_fn) pairs for the one-shot, process-wide
    OpenInference instrumentors (everything except FastAPI/ASGI, which is
    per-app -- see `_instrument_fastapi_app`).

    Imports happen lazily and per-instrumentor so that one package being
    absent or incompatible with the installed CrewAI/LlamaIndex/LiteLLM
    version can't prevent the others from being attempted. Each closure
    still takes only `tracer_provider` -- matching the pre-existing,
    tested one-argument contract (see
    `test_one_failing_instrumentor_does_not_disable_the_others`) -- and
    resolves the content-capture `TraceConfig` internally via
    `_trace_config()` rather than taking it as a second parameter.
    """

    def _crewai(tracer_provider):
        from openinference.instrumentation.crewai import CrewAIInstrumentor

        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider, config=_trace_config())

    def _llama_index(tracer_provider):
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(
            tracer_provider=tracer_provider, config=_trace_config()
        )

    def _litellm(tracer_provider):
        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider, config=_trace_config())

    def _openai(tracer_provider):
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=_trace_config())

    return [
        ("crewai", _crewai),
        ("llama_index", _llama_index),
        ("litellm", _litellm),
        ("openai", _openai),
    ]
