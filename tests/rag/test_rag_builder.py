from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.services.rag.rag_builder import BuildService

qdrant_store = pytest.importorskip(
    "app.services.vectorstore.qdrant_store",
    reason="Qdrant vector-store dependencies are not installed.",
)
QdrantService = qdrant_store.QdrantService


class FakeDocumentProcessor:
    def process_document(self, file_path: str | Path) -> list[dict[str, Any]]:
        return [
            {
                "text": "The warranty policy is valid for 14 days.",
                "metadata": {
                    "source": str(file_path),
                    "page_number": 1,
                    "chunk_index": 0,
                },
            },
            {
                "text": "Receipt code ALPHA-7 is required.",
                "metadata": {
                    "source": str(file_path),
                    "page_number": 2,
                    "chunk_index": 0,
                },
            },
        ]


class FakeEmbeddingService:
    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**chunk, "embedding": [0.1, 0.2, 0.3]} for chunk in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, Any, dict[str, Any] | None]] = []
        self.upserted_chunks: list[dict[str, Any]] = []

    def delete_by_filter(
        self,
        filter_field: str,
        filter_value: Any,
        exclude: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        self.deleted.append((filter_field, filter_value, exclude))
        return {"deleted_count": 1}

    def upsert(self, embedded_chunks: list[dict[str, Any]]) -> dict[str, int]:
        self.upserted_chunks = embedded_chunks
        return {"upserted": len(embedded_chunks), "failed": 0}


class _ArrayLike:
    def __init__(self, values: list[int] | list[float]) -> None:
        self._values = values

    def tolist(self) -> list[int] | list[float]:
        return self._values


class _SparseVector:
    indices = _ArrayLike([1, 2])
    values = _ArrayLike([0.4, 0.8])


class FakeSparseModel:
    def embed(self, texts: list[str]):
        for _ in texts:
            yield _SparseVector()


def test_build_embeds_and_upserts_text_chunks() -> None:
    vector_store = FakeVectorStore()
    service = BuildService(
        document_processor=FakeDocumentProcessor(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    summary = service.build_document("policy.pdf", source="policy.pdf")

    assert summary == {
        "source": "policy.pdf",
        "chunks_count": 2,
        "upserted": 2,
        "failed": 0,
    }
    assert len(vector_store.upserted_chunks) == 2
    first_metadata = vector_store.upserted_chunks[0]["metadata"]
    assert first_metadata["source"] == "policy.pdf"
    assert first_metadata["page_number"] == 1
    assert "page_image_path" not in first_metadata
    assert "page_image_url" not in first_metadata
    assert vector_store.deleted[0][0:2] == ("source", "policy.pdf")


def test_qdrant_point_payload_preserves_text_metadata() -> None:
    service = QdrantService.__new__(QdrantService)
    service.sparse_model = FakeSparseModel()

    points = service._build_points(
        [
            {
                "text": "Warranty instructions.",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": "policy.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "category": "warranty",
                },
            }
        ]
    )

    assert points[0].payload["text"] == "Warranty instructions."
    assert points[0].payload["category"] == "warranty"
    assert "page_image_url" not in points[0].payload
