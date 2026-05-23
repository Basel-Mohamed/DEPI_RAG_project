import pytest

from app.core.config import Settings
from app.core.startup import validate_startup_settings


def test_production_requires_auth_enabled():
    settings = Settings(DEPLOYMENT_ENV="production", AUTH_ENABLED=False, API_KEY="secret")

    with pytest.raises(RuntimeError, match="AUTH_ENABLED"):
        validate_startup_settings(settings)


def test_production_requires_api_key():
    settings = Settings(DEPLOYMENT_ENV="production", AUTH_ENABLED=True, API_KEY=None)

    with pytest.raises(RuntimeError, match="API_KEY"):
        validate_startup_settings(settings)


def test_development_allows_auth_disabled():
    settings = Settings(DEPLOYMENT_ENV="development", AUTH_ENABLED=False, API_KEY=None)

    validate_startup_settings(settings)
