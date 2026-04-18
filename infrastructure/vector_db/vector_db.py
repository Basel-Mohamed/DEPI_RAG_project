from typing import Optional

from domain.interfaces.vector_db_interface import VectorDBInterface
from domain.models.document import Document


class QdrantVectorDB(VectorDBInterface):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "Install 'qdrant-client' to use the Qdrant vector database."
                ) from exc

            self.client = QdrantClient(host=self.host, port=self.port)

        return self.client

    def _ensure_collection(self, vector_size: int):
        client = self._get_client()

        try:
            client.get_collection(self.collection_name)
            return
        except Exception:
            pass

        from qdrant_client.http.models import Distance, VectorParams

        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    @staticmethod
    def _build_payload(document: Document) -> dict:
        payload = dict(document.metadata)
        payload["text"] = document.page_content
        return payload

    @staticmethod
    def _build_filter(filter_payload: Optional[dict]):
        if not filter_payload:
            return None

        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_payload.items()
            ]
        )

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ):
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have a matching embedding.")

        client = self._get_client()
        self._ensure_collection(len(embeddings[0]))

        from qdrant_client.http.models import PointStruct

        points = [
            PointStruct(
                id=document.metadata["chunk_id"],
                vector=embedding,
                payload=self._build_payload(document),
            )
            for document, embedding in zip(documents, embeddings, strict=True)
        ]

        client.upsert(collection_name=self.collection_name, points=points)

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        filter: Optional[dict] = None,
    ):
        client = self._get_client()

        return client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=self._build_filter(filter),
            limit=top_k,
        )

    async def delete_documents(self, ids: list[str]):
        if not ids:
            return

        client = self._get_client()

        from qdrant_client.http.models import PointIdsList

        client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
