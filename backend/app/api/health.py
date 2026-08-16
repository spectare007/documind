import httpx
from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import HealthOut
from app.core.config import get_settings

router = APIRouter(tags=["health"])


def _check_url(url: str) -> bool:
    try:
        return httpx.get(url, timeout=3.0).status_code < 500
    except httpx.HTTPError:
        return False


def _check_db() -> bool:
    try:
        from app.db.session import get_engine
        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    s = get_settings()
    pg, ol, ph = _check_db(), _check_url(f"{s.ollama_base_url}/api/tags"), _check_url(s.phoenix_base_url)
    return HealthOut(status="ok" if (pg and ol) else "degraded", postgres=pg, ollama=ol, phoenix=ph)
