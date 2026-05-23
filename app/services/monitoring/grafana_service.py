from __future__ import annotations

import time
from threading import RLock
from typing import Any


class MetricsService:
    """Small in-process metrics registry for demo monitoring endpoints."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = RLock()
        self._counters: dict[str, int] = {
            "http_requests_total": 0,
            "http_errors_total": 0,
            "uploads_total": 0,
            "build_requests_total": 0,
            "build_failures_total": 0,
            "inference_requests_total": 0,
            "inference_errors_total": 0,
            "fallback_answers_total": 0,
            "retrieved_documents_total": 0,
        }
        self._latency: dict[str, list[float]] = {
            "http_request_ms": [],
            "inference_ms": [],
        }

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe_latency(self, name: str, value_ms: float) -> None:
        with self._lock:
            values = self._latency.setdefault(name, [])
            values.append(float(value_ms))
            if len(values) > 500:
                del values[:-500]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.started_at, 2),
                "counters": dict(self._counters),
                "latency": {
                    name: self._summarize(values)
                    for name, values in self._latency.items()
                },
            }

    def rag_summary(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        counters = snapshot["counters"]
        return {
            "uploads_total": counters.get("uploads_total", 0),
            "build_requests_total": counters.get("build_requests_total", 0),
            "build_failures_total": counters.get("build_failures_total", 0),
            "inference_requests_total": counters.get("inference_requests_total", 0),
            "inference_errors_total": counters.get("inference_errors_total", 0),
            "fallback_answers_total": counters.get("fallback_answers_total", 0),
            "retrieved_documents_total": counters.get("retrieved_documents_total", 0),
            "inference_latency_ms": snapshot["latency"].get("inference_ms", {}),
        }

    def prometheus_text(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP rag_app_uptime_seconds Process uptime in seconds.",
            "# TYPE rag_app_uptime_seconds gauge",
            f"rag_app_uptime_seconds {snapshot['uptime_seconds']}",
        ]
        for name, value in snapshot["counters"].items():
            metric_name = f"rag_app_{name}"
            lines.extend(
                [
                    f"# TYPE {metric_name} counter",
                    f"{metric_name} {value}",
                ]
            )
        for name, summary in snapshot["latency"].items():
            metric_base = f"rag_app_{name}"
            if summary["count"]:
                lines.extend(
                    [
                        f"# TYPE {metric_base}_count gauge",
                        f"{metric_base}_count {summary['count']}",
                        f"# TYPE {metric_base}_avg gauge",
                        f"{metric_base}_avg {summary['avg']}",
                        f"# TYPE {metric_base}_max gauge",
                        f"{metric_base}_max {summary['max']}",
                    ]
                )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "avg": None, "min": None, "max": None}
        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }


metrics_service = MetricsService()
