from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class MonitoringHealthResponse(BaseModel):
    status: str
    app: str
    version: str
    auth_enabled: bool
    qdrant_remote: bool


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    uptime_seconds: float
    counters: dict[str, int]
    latency: dict[str, Any]


class RagSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    uploads_total: int
    build_requests_total: int
    build_failures_total: int
    inference_requests_total: int
    inference_errors_total: int
    fallback_answers_total: int
    retrieved_documents_total: int
    inference_latency_ms: dict[str, Any]
