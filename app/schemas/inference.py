from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    include_sources: bool = True


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class SourcePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    id: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    answer: str
    content: list[ContentBlock] = Field(default_factory=list)
    sources: list[SourcePayload] = Field(default_factory=list)
    retrieval: dict[str, Any] = Field(default_factory=dict)


class InferenceStreamChunk(InferenceResponse):
    event: Literal["delta", "sources"] | str
