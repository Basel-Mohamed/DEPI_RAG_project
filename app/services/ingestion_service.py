from app.models.document import Document
from app.repositories.document_repository import InMemoryDocumentRepository
from app.repositories.vector_store import InMemoryVectorStore
from app.schemas.document import DocumentIngestRequest, DocumentIngestResponse


class IngestionService:
    def __init__(
        self,
        document_repository: InMemoryDocumentRepository,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self.document_repository = document_repository
        self.vector_store = vector_store

    def ingest_document(self, payload: DocumentIngestRequest) -> DocumentIngestResponse:
        document = Document(
            title=payload.title,
            content=payload.content,
            metadata=payload.metadata,
        )
        self.document_repository.save(document)
        self.vector_store.add_document(document)
        return DocumentIngestResponse(
            id=document.id,
            message="Document ingested successfully.",
        )

