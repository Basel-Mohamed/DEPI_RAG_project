from functools import lru_cache

from app.controllers.document_controller import DocumentController
from app.controllers.query_controller import QueryController
from app.repositories.document_repository import InMemoryDocumentRepository
from app.repositories.vector_store import InMemoryVectorStore
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


@lru_cache
def get_document_repository() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()


@lru_cache
def get_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


def get_ingestion_service() -> IngestionService:
    return IngestionService(
        document_repository=get_document_repository(),
        vector_store=get_vector_store(),
    )


def get_rag_service() -> RAGService:
    return RAGService(
        document_repository=get_document_repository(),
        vector_store=get_vector_store(),
    )


def get_document_controller() -> DocumentController:
    return DocumentController(ingestion_service=get_ingestion_service())


def get_query_controller() -> QueryController:
    return QueryController(rag_service=get_rag_service())

