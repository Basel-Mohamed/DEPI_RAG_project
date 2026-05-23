import pytest

from app.core.config import Settings

pytest.importorskip("fastembed")
pytest.importorskip("qdrant_client")

from app.services.vectorstore import qdrant_store
from app.services.vectorstore.qdrant_store import QdrantService


class FakeEmbeddingService:
    pass


class FakeSparseTextEmbedding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeQdrantClient:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeQdrantClient.last_kwargs = kwargs

    def collection_exists(self, collection):
        return True


def test_remote_qdrant_uses_https_url_and_api_key(monkeypatch):
    monkeypatch.setattr(qdrant_store, "SparseTextEmbedding", FakeSparseTextEmbedding)
    monkeypatch.setattr(qdrant_store, "QdrantClient", FakeQdrantClient)
    settings = Settings(
        QDRANT_REMOTE=True,
        QDRANT_HTTPS=True,
        QDRANT_HOST="example.qdrant.cloud",
        QDRANT_PORT=6333,
        QDRANT_API_KEY="qdrant-secret",
    )

    QdrantService(embedding_service=FakeEmbeddingService(), settings=settings)

    assert FakeQdrantClient.last_kwargs == {
        "url": "https://example.qdrant.cloud:6333",
        "api_key": "qdrant-secret",
        "prefer_grpc": False,
    }
