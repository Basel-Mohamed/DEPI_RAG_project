from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from app.services.llm.providers.base_llm import DEFAULT_FALLBACK_ANSWER

GeneratedPayload = dict[str, Any]


class GeneratedOutputNormalizer:
    """Normalize provider output into one text/multimodal response shape."""

    def normalize(self, payload: Any) -> GeneratedPayload:
        """Return ``answer``, ``content``, and ``images`` for any LLM output."""

        content: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        text_parts: list[str] = []

        self._collect_content(
            payload,
            content=content,
            images=images,
            text_parts=text_parts,
        )
        answer = "\n".join(part for part in text_parts if part).strip()
        if not content and answer:
            content.append({"type": "text", "text": answer})

        return {
            "answer": answer or DEFAULT_FALLBACK_ANSWER,
            "content": content,
            "images": images,
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

    def _collect_content(
        self,
        payload: Any,
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
        text_parts: list[str],
    ) -> None:
        """Recursively collect text and image blocks from a provider payload."""

        if payload is None:
            return
        if isinstance(payload, str):
            self._add_text(payload, content=content, text_parts=text_parts)
            return
        if isinstance(payload, bytes):
            self._add_image(
                {"data": base64.b64encode(payload).decode("ascii")},
                content=content,
                images=images,
            )
            return
        if isinstance(payload, Mapping):
            if self._try_collect_mapping(
                payload,
                content=content,
                images=images,
                text_parts=text_parts,
            ):
                return
            self._add_text(str(payload), content=content, text_parts=text_parts)
            return
        if isinstance(payload, Sequence):
            for item in payload:
                self._collect_content(
                    item,
                    content=content,
                    images=images,
                    text_parts=text_parts,
                )
            return
        if hasattr(payload, "content"):
            self._collect_content(
                getattr(payload, "content"),
                content=content,
                images=images,
                text_parts=text_parts,
            )
            return

        self._add_text(str(payload), content=content, text_parts=text_parts)

    def _try_collect_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
        text_parts: list[str],
    ) -> bool:
        """Handle dictionary-shaped provider payloads."""

        handled = False
        item_type = str(payload.get("type") or "").lower()

        if item_type in {"text", "output_text", "input_text"}:
            text = payload.get("text") or payload.get("content") or payload.get("value")
            if text is not None:
                self._add_text(str(text), content=content, text_parts=text_parts)
                handled = True

        if item_type in {"image", "image_url", "input_image", "output_image"}:
            self._add_image(payload, content=content, images=images)
            handled = True

        if handled and item_type:
            return True

        for key in ("answer", "text", "output_text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                self._add_text(value, content=content, text_parts=text_parts)
                handled = True
                break

        for key in ("content", "parts", "output"):
            if key in payload and not isinstance(payload.get(key), str):
                self._collect_content(
                    payload[key],
                    content=content,
                    images=images,
                    text_parts=text_parts,
                )
                handled = True

        for key in ("image", "images", "image_url", "image_urls", "media", "attachments"):
            if key in payload:
                self._collect_media_value(
                    payload[key],
                    content=content,
                    images=images,
                )
                handled = True

        if "b64_json" in payload or "base64" in payload or "data" in payload:
            self._add_image(payload, content=content, images=images)
            handled = True

        return handled

    @staticmethod
    def _add_text(
        text: str,
        *,
        content: list[dict[str, Any]],
        text_parts: list[str],
    ) -> None:
        """Append a non-empty text fragment to both answer and content lists."""

        clean_text = text.strip()
        if not clean_text:
            return
        text_parts.append(clean_text)
        content.append({"type": "text", "text": clean_text})

    def _collect_media_value(
        self,
        value: Any,
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> None:
        """Normalize nested media values from generated model output."""

        if value is None:
            return
        if isinstance(value, bytes):
            self._add_image(
                {"data": base64.b64encode(value).decode("ascii")},
                content=content,
                images=images,
            )
            return
        if isinstance(value, str):
            self._add_image({"url": value}, content=content, images=images)
            return
        if isinstance(value, Mapping):
            self._add_image(value, content=content, images=images)
            return
        if isinstance(value, Sequence):
            for item in value:
                self._collect_media_value(item, content=content, images=images)

    def _add_image(
        self,
        payload: Mapping[str, Any],
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> None:
        """Append an image block to normalized ``content`` and ``images`` lists."""

        image_url = payload.get("image_url")
        if isinstance(image_url, Mapping):
            image_url = image_url.get("url")

        image = {
            "type": "image",
            "url": image_url or payload.get("url"),
            "data": (
                payload.get("data")
                or payload.get("base64")
                or payload.get("b64_json")
                or payload.get("result")
            ),
            "mime_type": payload.get("mime_type") or payload.get("media_type"),
            "metadata": dict(payload.get("metadata") or {}),
        }
        image = {key: value for key, value in image.items() if value not in (None, {}, "")}
        if not image.get("url") and not image.get("data"):
            return
        images.append(image)
        content.append(image)
