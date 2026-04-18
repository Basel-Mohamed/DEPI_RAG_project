from domain.models.document import Document


class VectorStoreService:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ):
        await self.vector_db.add_documents(documents, embeddings)

    async def search(self, query_embedding, k: int = 5, filter: dict | None = None):
        return await self.vector_db.search(query_embedding, k, filter)

    async def delete_documents(self, ids: list[str]):
        await self.vector_db.delete_documents(ids)
