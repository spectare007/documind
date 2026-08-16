import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
