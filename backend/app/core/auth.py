import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Reject any request whose bearer token does not match `Settings.api_key`.

    Uses `hmac.compare_digest` rather than `!=` (fix for a review finding):
    a plain string comparison short-circuits on the first mismatched byte,
    so its runtime leaks information about how many leading characters of
    the guess were correct. `compare_digest` runs in time independent of
    where (or whether) the strings differ. Both operands are encoded to
    bytes first so the comparison is well-defined even if either string
    contains non-ASCII characters.
    """
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"), get_settings().api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
