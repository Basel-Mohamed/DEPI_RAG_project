from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from app.controllers.feedback_controller import FeedbackController
from app.services.metadata_store import get_metadata_store

logger = logging.getLogger(__name__)

REQUEST_LATENCY_MS = "request_latency_ms"
EMBEDDING_LATENCY_MS = "embedding_latency_ms"
RERANKING_LATENCY_MS = "reranking_latency_ms"
LLM_LATENCY_MS = "llm_latency_ms"
LLM_TOKENS = "llm_tokens"
QDRANT_LATENCY_MS = "qdrant_latency_ms"


class QdrantHealthProvider(Protocol):
    def health_check(self) -> dict[str, Any]:
        ...


class MonitoringMetrics:
    metrics_path = "monitoring/metrics.json"

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def record(self, metric_name: str, value: float) -> None:
        record = {
            "metric_id": str(uuid.uuid4()),
            "metric_name": metric_name,
            "value": max(float(value), 0.0),
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            try:
                get_metadata_store().append_monitoring_metric(
                    record,
                    self._metrics_path(),
                )
            except Exception:
                logger.exception("failed to persist monitoring metric=%s", metric_name)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            aggregates = get_metadata_store().aggregate_monitoring_metrics(
                self._metrics_path(),
            )

        request_latencies = aggregates.get(REQUEST_LATENCY_MS, {})
        llm_tokens = aggregates.get(LLM_TOKENS, {})
        llm_latencies = aggregates.get(LLM_LATENCY_MS, {})
        return {
            "request_count": int(request_latencies.get("count", 0)),
            "average_request_latency_ms": self._aggregate_average(request_latencies),
            "average_embedding_latency_ms": self._aggregate_average(
                aggregates.get(EMBEDDING_LATENCY_MS, {})
            ),
            "average_reranking_latency_ms": self._aggregate_average(
                aggregates.get(RERANKING_LATENCY_MS, {})
            ),
            "average_llm_latency_ms": self._aggregate_average(llm_latencies),
            "average_qdrant_latency_ms": self._aggregate_average(
                aggregates.get(QDRANT_LATENCY_MS, {})
            ),
            "llm_average_tokens_per_request": self._aggregate_average(llm_tokens),
            "llm_total_tokens": int(llm_tokens.get("total", 0)),
            "llm_request_count": int(llm_tokens.get("count", 0)),
        }

    def reset(self) -> None:
        with self._lock:
            get_metadata_store().clear_monitoring_metrics(self._metrics_path())

    @classmethod
    def record_request_latency(cls, latency_ms: float) -> None:
        monitoring_metrics.record(REQUEST_LATENCY_MS, latency_ms)

    @classmethod
    def record_embedding_latency(cls, latency_ms: float) -> None:
        monitoring_metrics.record(EMBEDDING_LATENCY_MS, latency_ms)

    @classmethod
    def record_reranking_latency(cls, latency_ms: float) -> None:
        monitoring_metrics.record(RERANKING_LATENCY_MS, latency_ms)

    @classmethod
    def record_llm(cls, latency_ms: float, tokens: int) -> None:
        monitoring_metrics.record(LLM_LATENCY_MS, latency_ms)
        monitoring_metrics.record(LLM_TOKENS, tokens)

    @classmethod
    def record_qdrant_latency(cls, latency_ms: float) -> None:
        monitoring_metrics.record(QDRANT_LATENCY_MS, latency_ms)

    @staticmethod
    def _aggregate_average(aggregate: dict[str, float | int]) -> float:
        if not aggregate:
            return 0.0
        return round(float(aggregate.get("average", 0.0)), 2)

    @classmethod
    def _metrics_path(cls):
        from pathlib import Path

        return Path.cwd() / cls.metrics_path


monitoring_metrics = MonitoringMetrics()
request_metrics = monitoring_metrics


class MonitoringController:
    def __init__(
        self,
        qdrant_service: QdrantHealthProvider,
        *,
        feedback_controller: FeedbackController | None = None,
        metrics: MonitoringMetrics = monitoring_metrics,
    ) -> None:
        self.qdrant_service = qdrant_service
        self.feedback_controller = feedback_controller or FeedbackController()
        self.metrics = metrics

    def metrics_summary(self) -> dict[str, Any]:
        snapshot = self.metrics.snapshot()
        return {
            **snapshot,
            "feedback_score": self.feedback_controller.satisfaction_score(),
        }

    def reset_metrics(self) -> dict[str, bool]:
        self.metrics.reset()
        return {"reset": True}

    def qdrant_health(self) -> dict[str, Any]:
        health = self.qdrant_service.health_check()
        if not health.get("healthy"):
            logger.warning("qdrant health check unhealthy error=%s", health.get("error"))
        return {
            "healthy": bool(health.get("healthy")),
            "collection": health.get("collection"),
            "status": health.get("status"),
            "points_count": health.get("points_count"),
            "files_count": health.get("files_count"),
            "error": health.get("error"),
        }
