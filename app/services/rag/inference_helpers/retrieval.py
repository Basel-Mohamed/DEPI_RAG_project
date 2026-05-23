from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.types import RetrievedContext


class RetrievalPolicy:
    """Rules for converting and selecting retrieved text context documents."""

    def effective_score_threshold(
        self,
        question: str,
        score_threshold: float | None,
    ) -> float | None:
        """Choose the search score threshold for a request."""

        return score_threshold

    @staticmethod
    def result_to_context(result: Mapping[str, Any]) -> RetrievedContext:
        """Convert a raw vector-store result into ``RetrievedContext``."""

        metadata = dict(result.get("metadata") or {})
        score = result.get("score")
        if score is not None:
            metadata["retrieval_score"] = score

        source = str(metadata.get("source") or "unknown source")
        page_number = metadata.get("page_number")
        title = str(metadata.get("title") or source)
        if page_number not in (None, ""):
            title = f"{title} p.{page_number}"

        return RetrievedContext(
            id=str(result.get("id") or metadata.get("id") or ""),
            title=title,
            content=str(result.get("text") or result.get("content") or ""),
            metadata=metadata,
        )
