from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.inference import get_inference_controller
from app.api.routes.monitoring import get_monitoring_controller
from app.controllers.feedback_controller import FeedbackController
from app.controllers.monitoring_controller import MonitoringController, MonitoringMetrics
from app.core.config import settings
from main import app


class FakeQdrantService:
    def health_check(self) -> dict[str, Any]:
        return {
            "healthy": True,
            "collection": "documents",
            "status": "green",
            "points_count": 7,
            "files_count": 2,
        }


class FakeInferenceController:
    def ask(self, request):
        return {
            "answer": "Answer.",
            "content": [{"type": "text", "text": "Answer."}],
            "sources": [],
            "retrieval": {"documents": 0},
        }

    def stream(self, request):
        yield '{"event":"delta","answer":"Answer.","content":[],"sources":[],"retrieval":{}}\n'


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(FeedbackController, "feedback_root", tmp_path / "feedback")
    monkeypatch.setattr(
        FeedbackController,
        "feedback_path",
        tmp_path / "feedback" / "feedback.json",
    )
    monkeypatch.setattr(MonitoringMetrics, "metrics_path", tmp_path / "metrics.json")
    MonitoringMetrics().reset()
    app.dependency_overrides[get_monitoring_controller] = lambda: MonitoringController(
        FakeQdrantService()
    )
    app.dependency_overrides[get_inference_controller] = lambda: FakeInferenceController()
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    MonitoringMetrics().reset()


def test_metrics_endpoint_returns_runtime_and_rag_kpis(client):
    client.get("/health")
    client.post(
        "/feedback",
        json={
            "session_id": "session-123",
            "question": "Question?",
            "answer": "Answer.",
            "rating": 1,
            "timestamp": "2026-05-23T15:30:00Z",
        },
    )
    client.post(
        "/feedback",
        json={
            "session_id": "session-123",
            "question": "Question?",
            "answer": "Answer.",
            "rating": -1,
            "timestamp": "2026-05-23T15:31:00Z",
        },
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] == 0
    assert body["average_request_latency_ms"] == 0
    assert body["average_embedding_latency_ms"] == 0
    assert body["average_reranking_latency_ms"] == 0
    assert body["average_llm_latency_ms"] == 0
    assert body["average_qdrant_latency_ms"] == 0
    assert body["feedback_score"] == 0.5
    assert body["llm_average_tokens_per_request"] == 0
    assert body["llm_total_tokens"] == 0
    assert body["llm_request_count"] == 0
    assert "qdrant_point_count" not in body
    assert "qdrant_file_count" not in body
    assert "qdrant_up" not in body


def test_metrics_request_count_tracks_only_question_requests(client):
    client.get("/health")
    client.get("/feedback")
    client.get("/metrics")

    before = client.get("/metrics").json()
    ask_response = client.post(
        "/ask",
        json={"question": "What is the refund policy?", "include_sources": True},
    )
    after = client.get("/metrics").json()

    assert ask_response.status_code == 200
    assert before["request_count"] == 0
    assert after["request_count"] == 1
    assert after["average_request_latency_ms"] >= 0


def test_metrics_request_count_tracks_stream_after_stream_finishes(client):
    response = client.post(
        "/ask/stream",
        json={"question": "What is the refund policy?", "include_sources": True},
    )
    body = client.get("/metrics").json()

    assert response.status_code == 200
    assert body["request_count"] == 1
    assert body["average_request_latency_ms"] >= 0


def test_qdrant_health_endpoint_returns_collection_status(client):
    response = client.get("/health/qdrant")

    assert response.status_code == 200
    assert response.json() == {
        "healthy": True,
        "collection": "documents",
        "status": "green",
        "points_count": 7,
        "files_count": 2,
        "error": None,
    }


def test_metrics_reset_endpoint_clears_persisted_metrics(client):
    client.get("/health")

    reset_response = client.post("/metrics/reset")
    metrics_response = client.get("/metrics")

    assert reset_response.status_code == 200
    assert reset_response.json() == {"reset": True}
    assert metrics_response.json()["request_count"] == 0


def test_monitoring_endpoints_require_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(MonitoringMetrics, "metrics_path", tmp_path / "metrics.json")
    monkeypatch.setattr(FeedbackController, "feedback_root", tmp_path / "feedback")
    monkeypatch.setattr(
        FeedbackController,
        "feedback_path",
        tmp_path / "feedback" / "feedback.json",
    )
    app.dependency_overrides[get_monitoring_controller] = lambda: MonitoringController(
        FakeQdrantService()
    )
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
