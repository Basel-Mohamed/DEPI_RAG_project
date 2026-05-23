from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def require_api_key(api_key: str | None = Header(default=None, alias=settings.API_KEY_HEADER)) -> None:
    """Protect production endpoints with a shared API key when enabled."""

    if not settings.AUTH_ENABLED:
        return
    if not settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is enabled but not configured.",
        )
    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
