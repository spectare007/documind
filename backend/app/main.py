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
    try:  # tracing + prompt sync are best-effort (real impls in Tasks 8-9)
        from app.observability.tracing import setup_tracing
        setup_tracing()
    except ImportError:
        logger.info("tracing module not present yet")
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
    app = FastAPI(
        title="DocuMind API",
        version="1.0.0",
        description="Agentic RAG document search platform",
        lifespan=lifespan,
    )
    from app.core.middleware import CorrelationIdMiddleware
    app.add_middleware(CorrelationIdMiddleware)
    from app.api import documents, health, ingest
    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    app.include_router(documents.router, dependencies=protected)
    app.include_router(ingest.router, dependencies=protected)
    return app


app = create_app()
