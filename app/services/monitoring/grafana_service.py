from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PrometheusExporter:
    def export(self, snapshot: dict[str, Any], feedback_summary: dict[str, Any]) -> str:
        """
        Convert a metrics snapshot + feedback summary into Prometheus text format.
        snapshot comes from MonitoringMetrics.snapshot().
        feedback_summary comes from FeedbackController.satisfaction_summary().
        Returns a UTF-8 string in Prometheus exposition format.
        """

        try:
            request_count = self._number(snapshot.get("request_count"))
            request_latency = self._number(snapshot.get("average_request_latency_ms"))
            embedding_latency = self._number(snapshot.get("average_embedding_latency_ms"))
            reranking_latency = self._number(snapshot.get("average_reranking_latency_ms"))
            llm_latency = self._number(snapshot.get("average_llm_latency_ms"))
            qdrant_latency = self._number(snapshot.get("average_qdrant_latency_ms"))
            llm_tokens = self._number(snapshot.get("llm_total_tokens"))
            feedback_total = self._number(feedback_summary.get("total"))
            feedback_positive = self._number(feedback_summary.get("positive"))
            satisfaction_score = self._number(feedback_summary.get("satisfaction_score"))
        except Exception:
            logger.exception("failed to convert metrics to prometheus values")
            request_count = 0.0
            request_latency = 0.0
            embedding_latency = 0.0
            reranking_latency = 0.0
            llm_latency = 0.0
            qdrant_latency = 0.0
            llm_tokens = 0.0
            feedback_total = 0.0
            feedback_positive = 0.0
            satisfaction_score = 0.0

        lines = [
            "# HELP rag_request_total Total number of inference requests",
            "# TYPE rag_request_total counter",
            f"rag_request_total {self._format_value(request_count)}",
            "",
            "# HELP rag_request_latency_ms_avg Average request latency in milliseconds",
            "# TYPE rag_request_latency_ms_avg gauge",
            f"rag_request_latency_ms_avg {self._format_value(request_latency)}",
            "",
            "# HELP rag_embedding_latency_ms_avg Average embedding latency in milliseconds",
            "# TYPE rag_embedding_latency_ms_avg gauge",
            f"rag_embedding_latency_ms_avg {self._format_value(embedding_latency)}",
            "",
            "# HELP rag_reranking_latency_ms_avg Average reranking latency in milliseconds",
            "# TYPE rag_reranking_latency_ms_avg gauge",
            f"rag_reranking_latency_ms_avg {self._format_value(reranking_latency)}",
            "",
            "# HELP rag_llm_latency_ms_avg Average LLM generation latency in milliseconds",
            "# TYPE rag_llm_latency_ms_avg gauge",
            f"rag_llm_latency_ms_avg {self._format_value(llm_latency)}",
            "",
            "# HELP rag_qdrant_latency_ms_avg Average Qdrant operation latency in milliseconds",
            "# TYPE rag_qdrant_latency_ms_avg gauge",
            f"rag_qdrant_latency_ms_avg {self._format_value(qdrant_latency)}",
            "",
            "# HELP rag_llm_tokens_total Total LLM tokens generated",
            "# TYPE rag_llm_tokens_total counter",
            f"rag_llm_tokens_total {self._format_value(llm_tokens)}",
            "",
            "# HELP rag_feedback_total Total feedback submissions",
            "# TYPE rag_feedback_total counter",
            f"rag_feedback_total {self._format_value(feedback_total)}",
            "",
            "# HELP rag_feedback_positive_total Total positive feedback submissions",
            "# TYPE rag_feedback_positive_total counter",
            f"rag_feedback_positive_total {self._format_value(feedback_positive)}",
            "",
            "# HELP rag_satisfaction_score User satisfaction score (0.0 to 1.0)",
            "# TYPE rag_satisfaction_score gauge",
            f"rag_satisfaction_score {self._format_value(satisfaction_score)}",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _number(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _format_value(value: float) -> str:
        try:
            number = float(value)
            if number.is_integer():
                return str(int(number))
            return f"{number:.6f}".rstrip("0").rstrip(".")
        except Exception:
            return "0"


class GrafanaDashboardBuilder:
    def __init__(self, title: str = "RAG System Monitoring", datasource: str = "Prometheus"):
        self.title = title
        self.datasource = datasource

    def build(self) -> dict[str, Any]:
        """Return a Grafana dashboard JSON dict ready to be POST-ed to Grafana API or saved as a file."""

        panels = [
            self._stat_panel(1, "Total Requests", "rag_request_total", 0, 0),
            self._gauge_panel(
                2,
                "Avg Request Latency (ms)",
                "rag_request_latency_ms_avg",
                12,
                0,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 1000},
                    {"color": "red", "value": 3000},
                ],
            ),
            self._gauge_panel(
                3,
                "Avg LLM Latency (ms)",
                "rag_llm_latency_ms_avg",
                0,
                8,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 2000},
                    {"color": "red", "value": 5000},
                ],
            ),
            self._gauge_panel(
                4,
                "User Satisfaction Score",
                "rag_satisfaction_score",
                12,
                8,
                min_value=0,
                max_value=1,
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.5},
                    {"color": "green", "value": 0.75},
                ],
            ),
            self._stat_panel(5, "Total LLM Tokens", "rag_llm_tokens_total", 0, 16),
            self._stat_panel(
                6,
                "Feedback Count",
                ["rag_feedback_total", "rag_feedback_positive_total"],
                12,
                16,
            ),
        ]
        return {
            "dashboard": {
                "uid": "rag-monitoring",
                "title": self.title,
                "schemaVersion": 38,
                "version": 1,
                "refresh": "30s",
                "time": {"from": "now-24h", "to": "now"},
                "panels": panels,
            },
            "overwrite": True,
        }

    def _stat_panel(
        self,
        panel_id: int,
        title: str,
        metric: str | list[str],
        x: int,
        y: int,
    ) -> dict[str, Any]:
        return {
            "id": panel_id,
            "type": "stat",
            "title": title,
            "datasource": {"type": "prometheus", "uid": self.datasource},
            "gridPos": {"x": x, "y": y, "w": 12, "h": 8},
            "targets": self._targets(metric),
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        }

    def _gauge_panel(
        self,
        panel_id: int,
        title: str,
        metric: str,
        x: int,
        y: int,
        *,
        thresholds: list[dict[str, float | str | None]],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> dict[str, Any]:
        field_config: dict[str, Any] = {
            "defaults": {
                "thresholds": {"mode": "absolute", "steps": thresholds},
            },
            "overrides": [],
        }
        if min_value is not None:
            field_config["defaults"]["min"] = min_value
        if max_value is not None:
            field_config["defaults"]["max"] = max_value
        return {
            "id": panel_id,
            "type": "gauge",
            "title": title,
            "datasource": {"type": "prometheus", "uid": self.datasource},
            "gridPos": {"x": x, "y": y, "w": 12, "h": 8},
            "targets": self._targets(metric),
            "fieldConfig": field_config,
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        }

    def _targets(self, metric: str | list[str]) -> list[dict[str, Any]]:
        metrics = [metric] if isinstance(metric, str) else metric
        return [
            {
                "expr": item,
                "legendFormat": item,
                "refId": chr(ord("A") + index),
            }
            for index, item in enumerate(metrics)
        ]


def generate_dashboard_json(output_path: str | Path) -> None:
    """Build the dashboard and write it to output_path as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(GrafanaDashboardBuilder().build(), indent=2),
        encoding="utf-8",
    )
