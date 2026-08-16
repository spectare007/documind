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
