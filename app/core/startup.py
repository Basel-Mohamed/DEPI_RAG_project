from __future__ import annotations

from app.core.config import Settings


def validate_startup_settings(settings: Settings) -> None:
    """Fail closed for production misconfiguration."""

    if settings.DEPLOYMENT_ENV.lower() != "production":
        return
    if not settings.AUTH_ENABLED:
        raise RuntimeError("AUTH_ENABLED must be true when DEPLOYMENT_ENV=production.")
    if not settings.API_KEY:
        raise RuntimeError("API_KEY is required when DEPLOYMENT_ENV=production.")
