from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from app.services.rag.media import MediaExtractor
from app.services.rag.output import GeneratedOutputNormalizer, GeneratedPayload
from app.services.types import RetrievedContext

SourcePayload = dict[str, Any]


class RagResponseBuilder:
    """Build normalized RAG responses and source metadata."""

    def __init__(
        self,
        *,
        media_extractor: MediaExtractor | None = None,
        output_normalizer: GeneratedOutputNormalizer | None = None,
    ) -> None:
        self.media_extractor = media_extractor or MediaExtractor()
        self.output_normalizer = output_normalizer or GeneratedOutputNormalizer()

    def build(
        self,
        raw_answer: Any,
        documents: list[RetrievedContext],
        *,
        retrieval: dict[str, Any],
        include_sources: bool = True,
    ) -> GeneratedPayload:
        """Normalize an LLM answer and attach source/retrieval metadata."""

        generated = self.output_normalizer.normalize(raw_answer)
        generated["sources"] = self.build_sources(documents) if include_sources else []
        generated["retrieval"] = retrieval
        return generated

    def build_delta(self, raw_chunk: Any) -> GeneratedPayload:
        """Normalize one streamed LLM chunk."""

        chunk = self.output_normalizer.normalize(raw_chunk)
        chunk["event"] = "delta"
        return chunk

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

        return self.output_normalizer.empty_response(answer)

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
