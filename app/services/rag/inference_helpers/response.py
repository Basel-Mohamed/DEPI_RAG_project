from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from app.services.llm.providers.base_llm import DEFAULT_FALLBACK_ANSWER
from app.services.media import MediaExtractor
from app.services.types import RetrievedContext

GeneratedPayload = dict[str, Any]
SourcePayload = dict[str, Any]


class RagResponseBuilder:
    """Build normalized RAG responses and source metadata."""

    def __init__(
        self,
        *,
        media_extractor: MediaExtractor | None = None,
    ) -> None:
        self.media_extractor = media_extractor or MediaExtractor()

    def build(
        self,
        raw_answer: Any,
        documents: list[RetrievedContext],
        *,
        retrieval: dict[str, Any],
        include_sources: bool = True,
    ) -> GeneratedPayload:
        """Normalize an LLM answer and attach source/retrieval metadata."""

        answer = str(raw_answer).strip() or DEFAULT_FALLBACK_ANSWER
        return {
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "sources": self.build_sources(documents) if include_sources else [],
            "retrieval": retrieval,
        }

    def build_delta(self, raw_chunk: Any) -> GeneratedPayload:
        """Normalize one streamed LLM chunk."""

        answer = str(raw_chunk).strip()
        return {
            "event": "delta",
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "sources": [],
            "retrieval": {},
        }

    def build_sources(self, documents: list[RetrievedContext]) -> list[SourcePayload]:
        """Build source payloads returned to API/CLI clients."""

        sources: list[SourcePayload] = []
        for index, document in enumerate(documents, start=1):
            sources.append(
                {
                    "rank": index,
                    "id": document.id,
                    "title": document.title,
                    "content": document.content,
                    "metadata": self.json_safe_metadata(document.metadata),
                    "media": self.media_extractor.extract(
                        document.metadata,
                        source_id=document.id,
                    ),
                }
            )
        return sources

    def empty_response(self, answer: str) -> GeneratedPayload:
        """Build the standard response shape for empty or fallback answers."""

        return {
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "sources": [],
            "retrieval": {"documents": 0},
        }

    @staticmethod
    def json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Return metadata that can be serialized to JSON."""

        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, bytes):
                safe[key] = base64.b64encode(value).decode("ascii")
            else:
                safe[key] = value
        return safe
