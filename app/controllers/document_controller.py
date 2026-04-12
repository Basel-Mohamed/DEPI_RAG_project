from app.schemas.document import DocumentIngestRequest, DocumentIngestResponse
from app.services.ingestion_service import IngestionService


class DocumentController:
    def __init__(self, ingestion_service: IngestionService) -> None:
        self.ingestion_service = ingestion_service

    def ingest(self, payload: DocumentIngestRequest) -> DocumentIngestResponse:
        return self.ingestion_service.ingest_document(payload)

