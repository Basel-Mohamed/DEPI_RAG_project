class VectorStoreService:

    def __init__(self, vector_db):
        self.vector_db = vector_db

    async def add_documents(self, ids, embeddings, metadata):

        await self.vector_db.add_documents(
            ids,
            embeddings,
            metadata
        )

    async def search(self, query_embedding, k=5):

        return await self.vector_db.search(
            query_embedding,
            k
        )