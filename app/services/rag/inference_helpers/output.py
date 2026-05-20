from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.services.llm.providers.base_llm import DEFAULT_FALLBACK_ANSWER

GeneratedPayload = dict[str, Any]


class GeneratedOutputNormalizer:
    """Normalize LLM output into the response text shape used by RAG."""

    def normalize(self, payload: Any) -> GeneratedPayload:
        """Return the final text answer and a text-only content list."""

        text_parts: list[str] = []
        self._collect_text(payload, text_parts=text_parts)

        answer = "\n".join(part for part in text_parts if part).strip()
        if not answer:
            answer = DEFAULT_FALLBACK_ANSWER

        return {
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "images": [],
            "sources": [],
            "retrieval": {},
        }

    @staticmethod
    def empty_response(answer: str = DEFAULT_FALLBACK_ANSWER) -> GeneratedPayload:
        """Build the standard response shape for fallback answers."""

        return {
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "images": [],
            "sources": [],
            "retrieval": {"documents": 0},
        }

    def _collect_text(self, payload: Any, *, text_parts: list[str]) -> None:
        """Recursively collect text fragments from a provider payload."""

        if payload is None:
            return
        if isinstance(payload, str):
            self._add_text(payload, text_parts=text_parts)
            return
        if isinstance(payload, bytes):
            return
        if isinstance(payload, Mapping):
            if self._try_collect_mapping(payload, text_parts=text_parts):
                return
            return
        if isinstance(payload, Sequence):
            for item in payload:
                self._collect_text(item, text_parts=text_parts)
            return
        if hasattr(payload, "content"):
            self._collect_text(getattr(payload, "content"), text_parts=text_parts)
            return

        self._add_text(str(payload), text_parts=text_parts)

    def _try_collect_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        text_parts: list[str],
    ) -> bool:
        """Handle dictionary-shaped provider payloads."""

        item_type = str(payload.get("type") or "").lower()
        if item_type in {"image", "image_url", "input_image", "output_image"}:
            return True

        if item_type in {"text", "output_text", "input_text"}:
            text = payload.get("text") or payload.get("content") or payload.get("value")
            if text is not None:
                self._add_text(str(text), text_parts=text_parts)
            return True

        for key in ("answer", "text", "output_text", "message", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                self._add_text(value, text_parts=text_parts)
                return True

        handled = False
        for key in ("content", "parts", "output"):
            if key in payload:
                self._collect_text(payload[key], text_parts=text_parts)
                handled = True

        return handled

    @staticmethod
    def _add_text(text: str, *, text_parts: list[str]) -> None:
        """Append a non-empty text fragment to the answer parts."""

        clean_text = text.strip()
        if clean_text:
            text_parts.append(clean_text)
