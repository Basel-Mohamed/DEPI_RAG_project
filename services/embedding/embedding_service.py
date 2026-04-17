class EmbeddingService:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def embed_documents(self, chunks):

        return await self.embedding_model.embed_documents(chunks)

    async def embed_query(self, query):

        return await self.embedding_model.embed_query(query)