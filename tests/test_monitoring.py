from app.controllers.monitoring_controller import MonitoringController
from app.services.monitoring.grafana_service import MetricsService


def test_metrics_snapshot_has_stable_shape():
    metrics = MetricsService()
    metrics.increment("uploads_total")
    metrics.observe_latency("inference_ms", 12.5)

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["uploads_total"] == 1
    assert snapshot["latency"]["inference_ms"]["count"] == 1


def test_monitoring_controller_returns_rag_summary():
    metrics = MetricsService()
    metrics.increment("inference_requests_total", 2)
    controller = MonitoringController(metrics)

    summary = controller.rag_summary()

    assert summary["inference_requests_total"] == 2
    assert "inference_latency_ms" in summary
