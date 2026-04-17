from qdrant_client import QdrantClient
from app.domain.interfaces.vector_db_interface import VectorDBInterface


class QdrantVectorDB(VectorDBInterface):

    def __init__(self):

        self.client = QdrantClient(host="localhost", port=6333)

    async def add_documents(self, ids, embeddings, metadata):

        self.client.upsert(
            collection_name="documents",
            points=[
                {
                    "id": ids[i],
                    "vector": embeddings[i],
                    "payload": metadata[i]
                }
                for i in range(len(ids))
            ]
        )

    async def search(self, embedding, top_k):

        results = self.client.search(
            collection_name="documents",
            query_vector=embedding,
            limit=top_k
        )

        return results