from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.rag.inference_helpers.media import MediaExtractor
from app.services.types import RetrievedContext


class RetrievalPolicy:
    """Rules for converting and selecting retrieved context documents."""

    def __init__(self, media_extractor: MediaExtractor | None = None) -> None:
        self.media_extractor = media_extractor or MediaExtractor()

    def effective_score_threshold(
        self,
        question: str,
        score_threshold: float | None,
    ) -> float | None:
        """Choose the search score threshold for a request."""

        if score_threshold is not None:
            return score_threshold
        if self.is_visual_question(question):
            return 0.0
        return None

    def ensure_visual_context(
        self,
        ranked_documents: list[RetrievedContext],
        retrieved_documents: list[RetrievedContext],
        *,
        top_k: int,
    ) -> list[RetrievedContext]:
        """Ensure visual questions include at least one media-bearing document."""

        if any(self.has_media(document) for document in ranked_documents):
            return ranked_documents

        media_document = next(
            (document for document in retrieved_documents if self.has_media(document)),
            None,
        )
        if media_document is None:
            return ranked_documents

        if len(ranked_documents) < top_k:
            return [*ranked_documents, media_document]
        if not ranked_documents:
            return [media_document]
        return [*ranked_documents[:-1], media_document]

    def has_media(self, document: RetrievedContext) -> bool:
        """Return whether a retrieved document contains image/media metadata."""

        return self.media_extractor.has_media(document.metadata, source_id=document.id)

    @staticmethod
    def is_visual_question(question: str) -> bool:
        """Heuristically detect questions that likely need image/table context."""

        lowered = question.lower()
        visual_terms = (
            "image",
            "picture",
            "photo",
            "figure",
            "diagram",
            "chart",
            "table",
            "schedule",
            "color",
            "colour",
            "shown",
            "look at",
            "visual",
            "screenshot",
            "column",
            "row",
            "الصورة",
            "صورة",
            "لون",
            "جدول",
            "عمود",
            "صف",
        )
        return any(term in lowered for term in visual_terms)

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
