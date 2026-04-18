from domain.models.document import Document


class EmbeddingService:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        texts = [document.page_content for document in documents]
        return await self.embedding_model.embed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        return await self.embedding_model.embed_query(query)
