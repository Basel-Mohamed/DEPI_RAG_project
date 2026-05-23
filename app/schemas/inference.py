from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    source: str | None = Field(
        default=None,
        description="Convenience filter for metadata.source.",
    )
    filter_field: str | None = None
    filter_value: Any = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    include_sources: bool = True

    @model_validator(mode="after")
    def validate_filter_options(self) -> "InferenceRequest":
        if self.source and (self.filter_field or self.filter_value is not None):
            raise ValueError("Use either source or filter_field/filter_value, not both.")
        if bool(self.filter_field) != (self.filter_value is not None):
            raise ValueError("filter_field and filter_value must be provided together.")
        return self


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
