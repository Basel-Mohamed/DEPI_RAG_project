import os
from collections import Counter

from infrastructure.embeddings.cohere_embeddings import CohereEmbeddings
from infrastructure.vector_db.vector_db import QdrantVectorDB
from services.embedding.embedding_service import EmbeddingService
from services.preprocessing.chunking_service import ChunkingService
from services.preprocessing.pdf_parser import PDFParser
from services.vector_db.vector_db import VectorStoreService


class RagBuildPipeline:
    def __init__(
        self,
        parser,
        chunker,
        embedding_service,
        vector_store,
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def run(self, file_path: str) -> dict:
        source_documents = self.parser.parse(file_path)
        chunked_documents = self.chunker.split(source_documents)

        if not chunked_documents:
            return {
                "document_count": 0,
                "chunk_count": 0,
                "modalities": {},
            }

        embeddings = await self.embedding_service.embed_documents(chunked_documents)
        await self.vector_store.add_documents(chunked_documents, embeddings)

        modality_counts = Counter(
            document.metadata.get("modality", "unknown")
            for document in chunked_documents
        )

        return {
            "document_count": len(source_documents),
            "chunk_count": len(chunked_documents),
            "modalities": dict(modality_counts),
            "document_id": chunked_documents[0].metadata.get("document_id"),
        }


def build_default_rag_build_pipeline() -> RagBuildPipeline:
    embedding_model = CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY", ""))

    return RagBuildPipeline(
        parser=PDFParser(),
        chunker=ChunkingService(),
        embedding_service=EmbeddingService(embedding_model=embedding_model),
        vector_store=VectorStoreService(vector_db=QdrantVectorDB()),
    )
