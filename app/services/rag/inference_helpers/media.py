from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any


class MediaExtractor:
    """Extract image/media metadata from retrieved RAG contexts."""

    def has_media(self, metadata: Mapping[str, Any], *, source_id: str = "") -> bool:
        """Return whether metadata contains at least one supported media item."""

        return bool(self.extract(metadata, source_id=source_id))

    def extract(
        self,
        metadata: Mapping[str, Any],
        *,
        source_id: str,
    ) -> list[dict[str, Any]]:
        """Return normalized media descriptors from document metadata.

        The builder stores PDF page images as ``page_image_base64`` data URLs,
        but this accepts URL fields and generic ``media``/``attachments`` too.
        """

        media: list[dict[str, Any]] = []
        for key in (
            "image",
            "images",
            "image_url",
            "image_urls",
            "image_base64",
            "page_image",
            "page_image_base64",
            "page_image_url",
            "media",
            "attachments",
        ):
            if key not in metadata:
                continue
            self._collect(metadata[key], media=media, source_id=source_id)
        return media

    def _collect(
        self,
        value: Any,
        *,
        media: list[dict[str, Any]],
        source_id: str,
    ) -> None:
        """Recursively normalize one media value from metadata."""

        if value is None:
            return
        if isinstance(value, bytes):
            media.append(
                {
                    "type": "image",
                    "data": base64.b64encode(value).decode("ascii"),
                    "source_id": source_id,
                }
            )
            return
        if isinstance(value, str):
            media.append({"type": "image", "url": value, "source_id": source_id})
            return
        if isinstance(value, Mapping):
            image: dict[str, Any] = {
                "type": value.get("type") or "image",
                "url": value.get("url") or value.get("image_url"),
                "data": value.get("data") or value.get("base64") or value.get("b64_json"),
                "mime_type": value.get("mime_type") or value.get("media_type"),
                "source_id": source_id,
            }
            media.append(
                {key: item for key, item in image.items() if item not in (None, "")}
            )
            return
        if isinstance(value, Sequence):
            for item in value:
                self._collect(item, media=media, source_id=source_id)
