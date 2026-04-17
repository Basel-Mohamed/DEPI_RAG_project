import cohere
from app.domain.interfaces.embedding_interface import EmbeddingInterface


class CohereEmbeddings(EmbeddingInterface):

    def __init__(self, api_key):

        self.client = cohere.Client(api_key)

    async def embed_documents(self, texts):

        response = self.client.embed(
            texts=texts,
            model="embed-english-v3.0"
        )

        return response.embeddings

    async def embed_query(self, query):

        response = self.client.embed(
            texts=[query],
            model="embed-english-v3.0"
        )

        return response.embeddings[0]