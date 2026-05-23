from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.rag.rag_builder import BuildService


class FakeDocumentProcessor:
    def process_document(self, file_path: str | Path) -> list[dict[str, Any]]:
        return [
            {
                "text": "The training schedule contains weekly sessions.",
                "metadata": {"source": str(file_path), "page_number": 1, "chunk_index": 0},
            }
        ]


class FakeEmbeddingService:
    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**chunk, "embedding": [0.1, 0.2, 0.3]} for chunk in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted_chunks: list[dict[str, Any]] = []

    def upsert(self, embedded_chunks: list[dict[str, Any]]) -> dict[str, int]:
        self.upserted_chunks = embedded_chunks
        return {"upserted": len(embedded_chunks), "failed": 0}

    def delete_by_filter(
        self,
        filter_field: str,
        filter_value: Any,
        exclude: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        return {"deleted_count": 0}


def test_build_service_upserts_text_chunks(tmp_path: Path) -> None:
    document_path = tmp_path / "training.pdf"
    document_path.write_bytes(b"%PDF-1.4 fake pdf bytes")
    vector_store = FakeVectorStore()
    service = BuildService(
        document_processor=FakeDocumentProcessor(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    result = service.build_document(document_path, source="file-123")

    metadata = vector_store.upserted_chunks[0]["metadata"]
    assert metadata["source"] == "file-123"
    assert "page_image_mime_type" not in metadata
    assert "page_image_base64" not in metadata
