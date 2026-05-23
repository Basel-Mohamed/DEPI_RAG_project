from __future__ import annotations

from pydantic import BaseModel, Field


class MetricsResponse(BaseModel):
    request_count: int = Field(..., ge=0)
    average_request_latency_ms: float = Field(..., ge=0)
    average_embedding_latency_ms: float = Field(..., ge=0)
    average_reranking_latency_ms: float = Field(..., ge=0)
    average_llm_latency_ms: float = Field(..., ge=0)
    average_qdrant_latency_ms: float = Field(..., ge=0)
    feedback_score: float | None = Field(
        default=None,
        description="Ratio of positive feedback items to total feedback items.",
    )
    llm_average_tokens_per_request: float = Field(..., ge=0)
    llm_total_tokens: int = Field(..., ge=0)
    llm_request_count: int = Field(..., ge=0)


class QdrantHealthResponse(BaseModel):
    healthy: bool
    collection: str | None = None
    status: str | None = None
    points_count: int | None = Field(default=None, ge=0)
    files_count: int | None = Field(default=None, ge=0)
    error: str | None = None


class MetricsResetResponse(BaseModel):
    reset: bool
