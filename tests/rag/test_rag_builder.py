from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.rag.rag_builder import RagBuildPipeline
from app.services.vectorstore.qdrant_store import QdrantService


class FakeDocumentProcessor:
    def process_document(self, file_path: str) -> tuple[list[dict[str, Any]], dict[int, bytes]]:
        return (
            [
                {
                    "text": "The warranty policy is valid for 14 days.",
                    "metadata": {
                        "source": file_path,
                        "page_number": 1,
                        "chunk_index": 0,
                    },
                },
                {
                    "text": "Receipt code ALPHA-7 is required.",
                    "metadata": {
                        "source": file_path,
                        "page_number": 2,
                        "chunk_index": 0,
                    },
                },
            ],
            {
                1: b"page-one-image",
                2: b"page-two-image",
            },
        )


class FakeEmbeddingService:
    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**chunk, "embedding": [0.1, 0.2, 0.3]} for chunk in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, Any]] = []
        self.upserted_chunks: list[dict[str, Any]] = []

    def delete_by_filter(self, filter_field: str, filter_value: Any) -> dict[str, int]:
        self.deleted.append((filter_field, filter_value))
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


def test_build_embeds_and_upserts_chunks_with_page_images() -> None:
    vector_store = FakeVectorStore()
    pipeline = RagBuildPipeline(
        document_processor=FakeDocumentProcessor(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        settings=SimpleNamespace(IMAGE_FORMAT="PNG"),
    )

    summary = pipeline.build("policy.pdf", replace_existing=True)

    assert summary == {
        "source": "policy.pdf",
        "chunks": 2,
        "page_images": 2,
        "upserted": 2,
        "failed": 0,
    }
    assert vector_store.deleted == [("source", "policy.pdf")]
    assert len(vector_store.upserted_chunks) == 2
    first_metadata = vector_store.upserted_chunks[0]["metadata"]
    assert first_metadata["page_image_mime_type"] == "image/png"
    assert first_metadata["page_image_base64"].startswith("data:image/png;base64,")


def test_build_can_skip_page_image_attachment() -> None:
    vector_store = FakeVectorStore()
    pipeline = RagBuildPipeline(
        document_processor=FakeDocumentProcessor(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        settings=SimpleNamespace(IMAGE_FORMAT="PNG"),
    )

    pipeline.build("policy.pdf", include_page_images=False)

    metadata = vector_store.upserted_chunks[0]["metadata"]
    assert "page_image_base64" not in metadata
    assert "page_image_mime_type" not in metadata


def test_qdrant_point_payload_preserves_extra_metadata() -> None:
    service = QdrantService.__new__(QdrantService)
    service.sparse_model = FakeSparseModel()

    points = service._build_points(
        [
            {
                "text": "Visual warranty instructions.",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": "policy.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "page_image_base64": "data:image/png;base64,abc123",
                },
            }
        ]
    )

    assert points[0].payload["page_image_base64"] == "data:image/png;base64,abc123"
