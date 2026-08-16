import uuid

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.correlation import set_correlation_id


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Propagates a request correlation ID through logs, spans, and responses.

    Reads `X-Correlation-ID` from the incoming request (or generates a uuid4
    hex if absent), stores it in the `app.core.correlation` contextvar so log
    records and downstream code can read it, stamps it as a `correlation.id`
    attribute on the current OTel span if one is recording, and echoes it
    back on the response header so callers can correlate their own logs.
    """

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        set_correlation_id(cid)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("correlation.id", cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response
