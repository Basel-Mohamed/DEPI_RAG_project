import hmac

from fastapi import Header, HTTPException, status

from app.core.config import settings


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    configured_api_key = settings.API_KEY
    if not configured_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication is not configured.",
        )

    provided_api_key = x_api_key or _bearer_token(authorization)
    if not provided_api_key or not hmac.compare_digest(provided_api_key, configured_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        return token.strip() or None
    return None
