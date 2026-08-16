import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_prompt_manager_cache():
    from app.observability.prompts import get_prompt_manager
    get_prompt_manager.cache_clear()
    yield
    get_prompt_manager.cache_clear()
