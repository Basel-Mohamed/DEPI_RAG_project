from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, settings as global_settings

if TYPE_CHECKING:
    from app.services.embedding.embedding_service import EmbeddingService
    from app.services.preprocessing.preprocessing_service import DocumentProcessor
    from app.services.vectorstore.qdrant_store import QdrantService

logger = logging.getLogger(__name__)


class RagBuilderError(RuntimeError):
    """Raised when the RAG build pipeline cannot index a document."""


class RagBuildPipeline:
    """Process a source document, embed its chunks, and store them in Qdrant."""

    def __init__(
        self,
        *,
        document_processor: DocumentProcessor | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or global_settings

        if document_processor is None:
            from app.services.preprocessing.preprocessing_service import DocumentProcessor

            document_processor = DocumentProcessor(settings=self.settings)

        if embedding_service is None:
            from app.services.embedding.embedding_service import EmbeddingService

            embedding_service = EmbeddingService(settings=self.settings)

        if vector_store is None:
            from app.services.vectorstore.qdrant_store import QdrantService

            vector_store = QdrantService(
                embedding_service=embedding_service,
                settings=self.settings,
            )

        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def build(
        self,
        file_path: str | Path,
        *,
        include_page_images: bool = True,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """Build and index one document.

        Args:
            file_path: Source document path accepted by ``DocumentProcessor``.
            include_page_images: Attach extracted page images to matching chunks
                as base64 data URLs so multimodal inference can pass them to a
                vision-capable model.
            replace_existing: Delete existing chunks for the same source before
                upserting the new ones.
        """

        try:
            chunks, page_images = self.document_processor.process_document(file_path)
        except Exception as exc:
            raise RagBuilderError(f"Document processing failed for {file_path}.") from exc

        if include_page_images:
            chunks = self._attach_page_images(chunks, page_images)

        if not chunks:
            logger.warning("No chunks produced for %s.", file_path)
            return {
                "source": str(file_path),
                "chunks": 0,
                "page_images": len(page_images),
                "upserted": 0,
                "failed": 0,
            }

        source = str(chunks[0].get("metadata", {}).get("source") or file_path)

        if replace_existing:
            try:
                self.vector_store.delete_by_filter("source", source)
            except Exception as exc:
                raise RagBuilderError(
                    f"Failed to delete existing chunks for {source}."
                ) from exc

        try:
            embedded_chunks = self.embedding_service.embed_chunks(chunks)
        except Exception as exc:
            raise RagBuilderError(f"Embedding failed for {source}.") from exc

        try:
            result = self.vector_store.upsert(embedded_chunks)
        except Exception as exc:
            raise RagBuilderError(f"Vector store upsert failed for {source}.") from exc

        summary = {
            "source": source,
            "chunks": len(chunks),
            "page_images": len(page_images),
            "upserted": int(result.get("upserted", 0)),
            "failed": int(result.get("failed", 0)),
        }
        logger.info("Built RAG index for %s: %s", source, summary)
        return summary

    def build_many(
        self,
        file_paths: list[str | Path],
        *,
        include_page_images: bool = True,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """Build and index multiple documents."""

        results = [
            self.build(
                file_path,
                include_page_images=include_page_images,
                replace_existing=replace_existing,
            )
            for file_path in file_paths
        ]
        return {
            "documents": len(results),
            "chunks": sum(item["chunks"] for item in results),
            "page_images": sum(item["page_images"] for item in results),
            "upserted": sum(item["upserted"] for item in results),
            "failed": sum(item["failed"] for item in results),
            "results": results,
        }

    def close(self) -> None:
        """Release resources held by the vector store."""
        close = getattr(self.vector_store, "close", None)
        if callable(close):
            close()

    def _attach_page_images(
        self,
        chunks: list[dict[str, Any]],
        page_images: dict[int, bytes],
    ) -> list[dict[str, Any]]:
        if not page_images:
            return chunks

        image_format = str(getattr(self.settings, "IMAGE_FORMAT", "PNG")).lower()
        mime_type = self._mime_type(image_format)
        encoded_images = {
            page_number: self._encode_image_data_url(image_bytes, mime_type=mime_type)
            for page_number, image_bytes in page_images.items()
        }

        enriched_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            page_number = metadata.get("page_number")
            page_image = encoded_images.get(page_number)
            if page_image:
                metadata.update(
                    {
                        "page_image_base64": page_image,
                        "page_image_mime_type": mime_type,
                    }
                )

            enriched_chunks.append(
                {
                    **chunk,
                    "metadata": metadata,
                }
            )

        return enriched_chunks

    @staticmethod
    def _encode_image_data_url(image_bytes: bytes, *, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _mime_type(image_format: str) -> str:
        normalized = image_format.lower()
        if normalized == "jpg":
            normalized = "jpeg"
        return f"image/{normalized}"


RagBuilderPipeline = RagBuildPipeline
RagBuilderService = RagBuildPipeline
