from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from app.services.llm.providers.base_llm import DEFAULT_FALLBACK_ANSWER
from app.services.rag.inference_helpers.media import MediaExtractor
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

        answer = self.answer_text(raw_answer)
        return {
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "sources": self.build_sources(documents) if include_sources else [],
            "retrieval": retrieval,
        }

    def build_delta(self, raw_chunk: Any) -> GeneratedPayload:
        """Normalize one streamed LLM chunk."""

        answer = self.answer_text(raw_chunk, fallback="")
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

    def answer_text(
        self,
        payload: Any,
        *,
        fallback: str = DEFAULT_FALLBACK_ANSWER,
    ) -> str:
        """Extract the text answer from an LLM provider payload."""

        text_parts: list[str] = []
        self._collect_text(payload, text_parts=text_parts)
        return "\n".join(text_parts).strip() or fallback

    def _collect_text(self, payload: Any, *, text_parts: list[str]) -> None:
        """Collect text from strings, messages, and simple content dictionaries."""

        if payload is None or isinstance(payload, bytes):
            return
        if isinstance(payload, str):
            self._add_text(payload, text_parts=text_parts)
            return
        if hasattr(payload, "content"):
            self._collect_text(getattr(payload, "content"), text_parts=text_parts)
            return
        if isinstance(payload, Mapping):
            item_type = str(payload.get("type") or "").lower()
            if item_type in {"image", "image_url", "input_image", "output_image"}:
                return
            for key in ("answer", "text", "output_text", "message", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    self._add_text(value, text_parts=text_parts)
                    return
            for key in ("content", "parts", "output"):
                if key in payload:
                    self._collect_text(payload[key], text_parts=text_parts)
            return
        if isinstance(payload, Sequence):
            for item in payload:
                self._collect_text(item, text_parts=text_parts)
            return
        self._add_text(str(payload), text_parts=text_parts)

    @staticmethod
    def _add_text(text: str, *, text_parts: list[str]) -> None:
        clean_text = text.strip()
        if clean_text:
            text_parts.append(clean_text)

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
