from pydantic import BaseModel
from typing import Any


class HealthResponse(BaseModel):
    status: str
    environment: str | None = None


class MetricSample(BaseModel):
    value: float
    labels: dict[str, str]


class StatsResponse(BaseModel):
    qdrant_up: bool
    metrics: dict[str, MetricSample]