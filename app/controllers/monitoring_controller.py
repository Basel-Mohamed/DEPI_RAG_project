from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.services.monitoring.grafana_service import MetricsService, metrics_service


class MonitoringController:
    def __init__(self, metrics: MetricsService = metrics_service) -> None:
        self.metrics = metrics

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "auth_enabled": settings.AUTH_ENABLED,
            "qdrant_remote": settings.QDRANT_REMOTE,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def rag_summary(self) -> dict[str, Any]:
        return self.metrics.rag_summary()

    def prometheus(self) -> str:
        return self.metrics.prometheus_text()
