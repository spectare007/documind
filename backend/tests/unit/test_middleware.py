from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app():
    from app.core.middleware import CorrelationIdMiddleware
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping():
        from app.core.correlation import get_correlation_id
        return {"cid": get_correlation_id()}

    return app


def test_generates_correlation_id():
    r = TestClient(_app()).get("/ping")
    assert r.headers["x-correlation-id"]
    assert r.json()["cid"] == r.headers["x-correlation-id"]


def test_respects_incoming_correlation_id():
    r = TestClient(_app()).get("/ping", headers={"X-Correlation-ID": "abc-123"})
    assert r.headers["x-correlation-id"] == "abc-123"
    assert r.json()["cid"] == "abc-123"


def test_correlation_id_lands_on_a_real_recording_span():
    """Proves the `correlation.id` attribute reaches an actual exported span.

    Regression test for a review finding: `CorrelationIdMiddleware` reads
    `trace.get_current_span()`, which is only ever a non-recording no-op
    span unless some ASGI/FastAPI OTel instrumentor has created a real
    server span for the request first (see `app.observability.tracing`,
    which wires up `FastAPIInstrumentor.instrument_app(app, ...)` for
    exactly this reason). This test instruments a throwaway app with a
    dedicated in-memory exporter -- no network, no global tracer state --
    and asserts the exported span actually carries the attribute the
    middleware set, not just that the middleware ran without erroring.
    """
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from app.core.middleware import CorrelationIdMiddleware

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    try:
        r = TestClient(app).get("/ping", headers={"X-Correlation-ID": "trace-me-123"})
        assert r.status_code == 200
    finally:
        FastAPIInstrumentor.uninstrument_app(app)

    server_spans = [s for s in exporter.get_finished_spans() if s.name == "GET /ping"]
    assert len(server_spans) == 1
    assert server_spans[0].attributes.get("correlation.id") == "trace-me-123"
