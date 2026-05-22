import base64
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from app.services.embedding.embedding_service import EmbeddingService
    from app.services.preprocessing.preprocessing_service import DocumentProcessor
    from app.services.vectorstore.qdrant_store import QdrantService

logger = logging.getLogger(__name__)


class BuildService:
    """Build and clean vector indexes from source documents."""

    def __init__(
        self,
        document_processor: "DocumentProcessor",
        embedding_service: "EmbeddingService",
        vector_store: "QdrantService",
    ) -> None:
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def build_document(self, file_path: str | Path, source: str | None = None) -> dict[str, Any]:
        file_path = Path(file_path)
        normalized_source = source or Path(file_path).name

        logger.info(
            "build started source=%s file_path=%s",
            normalized_source,
            file_path,
        )

        logger.info("document processing started source=%s", normalized_source)
        chunks, page_images = self.document_processor.process_document(file_path)
        logger.info(
            "document processing completed source=%s chunks_count=%s page_images_count=%s",
            normalized_source,
            len(chunks),
            len(page_images),
        )

        logger.info("metadata normalization started source=%s", normalized_source)
        build_id = uuid.uuid4().hex
        chunks_with_page_images_count = 0
        for chunk in chunks:
            metadata = chunk.setdefault("metadata", {})
            metadata["source"] = normalized_source
            metadata["build_id"] = build_id
            if self._attach_page_image(metadata, page_images):
                chunks_with_page_images_count += 1
            self._validate_chunk(chunk)
        logger.info(
            "metadata normalization completed source=%s chunks_with_page_images_count=%s",
            normalized_source,
            chunks_with_page_images_count,
        )

        logger.info("embedding started source=%s chunks_count=%s", normalized_source, len(chunks))
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        if len(embedded_chunks) != len(chunks):
            raise RuntimeError(
                f"Embedding service returned {len(embedded_chunks)} chunks for {len(chunks)} inputs."
            )
        logger.info(
            "embedding completed source=%s embedded_chunks_count=%s",
            normalized_source,
            len(embedded_chunks),
        )

        logger.info("vector upsert started source=%s chunks_count=%s", normalized_source, len(embedded_chunks))
        upsert_result = self.vector_store.upsert(embedded_chunks)
        logger.info(
            "vector upsert completed source=%s upserted=%s failed=%s",
            normalized_source,
            upsert_result["upserted"],
            upsert_result["failed"],
        )

        stale_cleanup_result = self.vector_store.delete_by_filter(
            "source",
            normalized_source,
            exclude={"build_id": build_id},
        )
        logger.info(
            "stale chunk cleanup completed source=%s build_id=%s deleted_count=%s",
            normalized_source,
            build_id,
            stale_cleanup_result["deleted_count"],
        )

        result = {
            "source": normalized_source,
            "chunks_count": len(chunks),
            "page_images_count": len(page_images),
            "chunks_with_page_images_count": chunks_with_page_images_count,
            "upserted": upsert_result["upserted"],
            "failed": upsert_result["failed"],
        }
        logger.info("build completed source=%s result=%s", normalized_source, result)
        return result

    def get_document_status(self, source: str) -> dict[str, Any]:
        logger.info("build status lookup started source=%s", source)
        chunks_count = self.vector_store.count_by_filter("source", source)
        result = {"source": source, "chunks_count": chunks_count}
        logger.info("build status lookup completed source=%s chunks_count=%s", source, chunks_count)
        return result

    def delete_document(self, source: str) -> dict[str, Any]:
        logger.info("build cleanup started source=%s", source)
        result = self.vector_store.delete_by_filter("source", source)
        cleanup_result = {"source": source, "deleted_count": result["deleted_count"]}
        logger.info(
            "build cleanup completed source=%s deleted_count=%s",
            source,
            result["deleted_count"],
        )
        return cleanup_result

    @staticmethod
    def _validate_chunk(chunk: dict[str, Any]) -> None:
        if not chunk.get("text"):
            raise ValueError("Document processor returned a chunk without text.")
        metadata = chunk.get("metadata") or {}
        for key in ("source", "page_number", "chunk_index", "build_id"):
            if key not in metadata:
                raise ValueError(f"Document processor returned a chunk without metadata.{key}.")

    def _attach_page_image(
        self,
        metadata: dict[str, Any],
        page_images: dict[int, bytes],
    ) -> bool:
        page_number = metadata.get("page_number")
        if page_number is None:
            return False

        try:
            normalized_page_number = int(page_number)
        except (TypeError, ValueError):
            return False

        image_bytes = page_images.get(normalized_page_number)
        if not image_bytes:
            return False

        image_format = str(
            getattr(self.document_processor, "image_format", "PNG")
        ).lower()
        image_subtype = "jpeg" if image_format == "jpg" else image_format
        metadata["page_image_mime_type"] = f"image/{image_subtype}"
        metadata["page_image_base64"] = (
            f"data:{metadata['page_image_mime_type']};base64,"
            f"{base64.b64encode(image_bytes).decode('ascii')}"
        )
        return True
