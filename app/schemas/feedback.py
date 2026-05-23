from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: Literal[1, -1]
    timestamp: datetime


class FeedbackResponse(BaseModel):
    feedback_id: str
    session_id: str
    stored: bool


class FeedbackRecord(BaseModel):
    feedback_id: str
    session_id: str
    question: str
    answer: str
    rating: Literal[1, -1]
    timestamp: datetime
    created_at: datetime


class FeedbackListResponse(BaseModel):
    feedback: list[FeedbackRecord]
    total: int


class FeedbackSatisfactionResponse(BaseModel):
    total: int
    positive: int
    negative: int
    satisfaction_score: float | None
    satisfaction_percent: float | None
