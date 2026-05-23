from __future__ import annotations

from typing import Any

from app.controllers.monitoring_controller import (
    LLM_TOKENS,
    MonitoringController,
    MonitoringMetrics,
    REQUEST_LATENCY_MS,
)
from app.core.config import settings


class FakeQdrantService:
    def __init__(self, health: dict[str, Any]) -> None:
        self.health = health
        self.calls = 0

    def health_check(self) -> dict[str, Any]:
        self.calls += 1
        return self.health


class FakeFeedbackController:
    def __init__(self, score: float | None) -> None:
        self.score = score

    def satisfaction_score(self) -> float | None:
        return self.score


def test_metrics_summary_combines_request_feedback_without_qdrant_lookup(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(MonitoringMetrics, "metrics_path", tmp_path / "metrics.json")
    metrics = MonitoringMetrics()
    metrics.record(REQUEST_LATENCY_MS, 100.0)
    metrics.record(REQUEST_LATENCY_MS, 200.0)
    metrics.record(LLM_TOKENS, 30)
    qdrant_service = FakeQdrantService(
        {
            "healthy": True,
            "collection": "documents",
            "status": "green",
            "points_count": 42,
            "files_count": 3,
        }
    )
    controller = MonitoringController(
        qdrant_service,
        feedback_controller=FakeFeedbackController(score=0.75),
        metrics=metrics,
    )

    assert controller.metrics_summary() == {
        "request_count": 2,
        "average_request_latency_ms": 150.0,
        "average_embedding_latency_ms": 0.0,
        "average_reranking_latency_ms": 0.0,
        "average_llm_latency_ms": 0.0,
        "average_qdrant_latency_ms": 0.0,
        "feedback_score": 0.75,
        "llm_average_tokens_per_request": 30.0,
        "llm_total_tokens": 30,
        "llm_request_count": 1,
    }
    assert qdrant_service.calls == 0


def test_qdrant_health_preserves_unhealthy_error_details(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(MonitoringMetrics, "metrics_path", tmp_path / "metrics.json")
    controller = MonitoringController(
        FakeQdrantService({"healthy": False, "error": "connection refused"}),
        feedback_controller=FakeFeedbackController(score=None),
        metrics=MonitoringMetrics(),
    )

    assert controller.qdrant_health() == {
        "healthy": False,
        "collection": None,
        "status": None,
        "points_count": None,
        "files_count": None,
        "error": "connection refused",
    }


def test_reset_metrics_clears_persisted_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(MonitoringMetrics, "metrics_path", tmp_path / "metrics.json")
    metrics = MonitoringMetrics()
    metrics.record(REQUEST_LATENCY_MS, 100.0)
    controller = MonitoringController(
        FakeQdrantService({"healthy": True, "points_count": 0}),
        feedback_controller=FakeFeedbackController(score=None),
        metrics=metrics,
    )

    assert controller.reset_metrics() == {"reset": True}
    assert controller.metrics_summary()["request_count"] == 0
