import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.core.auth import require_api_key
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    from app.db.session import init_db
    init_db()
    try:
        from app.observability.prompts import get_prompt_manager
        manager = get_prompt_manager()
        manager.sync_to_phoenix()
        # Pull back any UI edits so they win over YAML for this process,
        # without needing a restart (see PromptManager.refresh_from_phoenix).
        manager.refresh_from_phoenix()
    except ImportError:
        logger.info("prompt manager not present yet")
    yield


def create_app() -> FastAPI:
    # Configured here (idempotently -- see `setup_logging`) rather than only
    # in `lifespan()`, so `setup_tracing()` below has a real handler to log
    # through instead of silently losing its INFO-level summary to Python's
    # unconfigured-root-logger last-resort handler (WARNING+ only).
    setup_logging()
    app = FastAPI(
        title="DocuMind API",
        version="1.0.0",
        description="Agentic RAG document search platform",
        lifespan=lifespan,
    )
    from app.core.middleware import CorrelationIdMiddleware
    app.add_middleware(CorrelationIdMiddleware)
    try:  # tracing is best-effort (real impl in Task 8)
        # Must run here -- synchronously, before this function returns --
        # rather than inside `lifespan()`. Starlette caches
        # `self.middleware_stack` on the app's very first ASGI call, which
        # for uvicorn *is* the lifespan-startup call; instrumenting FastAPI
        # from inside `lifespan()` (as an earlier version of this code did)
        # patches `build_middleware_stack` after Starlette already built and
        # cached the uninstrumented stack, so it silently never takes effect
        # for any real request even though instrumentation "succeeds" with
        # no error. Calling it here, before `app` is ever invoked, avoids
        # that trap.
        from app.observability.tracing import setup_tracing
        setup_tracing(app)
    except ImportError:
        logger.info("tracing module not present yet")
    from app.api import documents, health, ingest, openai_compat, query
    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    app.include_router(documents.router, dependencies=protected)
    app.include_router(ingest.router, dependencies=protected)
    app.include_router(query.router, dependencies=protected)
    app.include_router(openai_compat.router, dependencies=protected)
    return app


app = create_app()
