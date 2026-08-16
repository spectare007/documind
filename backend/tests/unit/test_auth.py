"""Tests for `app.core.auth.require_api_key`.

Covers the review-finding fix: the bearer-token check now uses
`hmac.compare_digest` instead of a plain `!=` comparison, which is
vulnerable to timing attacks (an attacker who can measure response latency
can recover the secret one byte at a time from a naive short-circuiting
comparison).
"""

import hmac
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import require_api_key
from app.core.config import Settings

_SETTINGS = Settings(_env_file=None, api_key="documind-dev-key")


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_valid_key_passes():
    with patch("app.core.auth.get_settings", return_value=_SETTINGS):
        require_api_key(_creds("documind-dev-key"))  # must not raise


def test_invalid_key_rejected():
    with patch("app.core.auth.get_settings", return_value=_SETTINGS):
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(_creds("wrong-key"))
    assert exc_info.value.status_code == 401


def test_missing_credentials_rejected():
    with patch("app.core.auth.get_settings", return_value=_SETTINGS):
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(None)
    assert exc_info.value.status_code == 401


def test_key_comparison_uses_hmac_compare_digest(monkeypatch):
    """Regression test for the review finding: this must call
    `hmac.compare_digest`, not Python's `!=`, so the comparison runs in
    constant time regardless of where (or whether) the strings differ.
    """
    invoked_with = {}
    real_compare_digest = hmac.compare_digest

    def spy(a, b):
        invoked_with["args"] = (a, b)
        return real_compare_digest(a, b)

    monkeypatch.setattr("app.core.auth.hmac.compare_digest", spy)

    with patch("app.core.auth.get_settings", return_value=_SETTINGS):
        require_api_key(_creds("documind-dev-key"))

    assert invoked_with.get("args") == (b"documind-dev-key", b"documind-dev-key")


def test_key_comparison_still_rejects_wrong_key_when_instrumented(monkeypatch):
    """The spy above proves `compare_digest` is called; this proves the real
    (unspied) `compare_digest` path still correctly rejects a wrong key, so
    the fix did not accidentally weaken the check.
    """
    with patch("app.core.auth.get_settings", return_value=_SETTINGS):
        with pytest.raises(HTTPException):
            require_api_key(_creds("documind-dev-ke"))  # one char short
        with pytest.raises(HTTPException):
            require_api_key(_creds("documind-dev-keyX"))  # one char extra
