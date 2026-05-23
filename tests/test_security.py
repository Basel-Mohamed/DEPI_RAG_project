from fastapi.testclient import TestClient

from app.core.config import settings
from main import app


def test_health_is_public_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "API_KEY", "secret")

    response = TestClient(app).get("/health")

    assert response.status_code == 200


def test_protected_endpoint_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "API_KEY", "secret")

    response = TestClient(app).get("/monitoring/health")

    assert response.status_code == 401


def test_protected_endpoint_accepts_api_key(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "API_KEY", "secret")

    response = TestClient(app).get("/monitoring/health", headers={"X-API-Key": "secret"})

    assert response.status_code == 200
