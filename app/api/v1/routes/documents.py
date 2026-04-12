from fastapi import APIRouter, Depends, status

from app.controllers.document_controller import DocumentController
from app.dependencies import get_document_controller
from app.schemas.document import DocumentIngestRequest, DocumentIngestResponse


router = APIRouter()


@router.post(
    "",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_document(
    payload: DocumentIngestRequest,
    controller: DocumentController = Depends(get_document_controller),
) -> DocumentIngestResponse:
    return controller.ingest(payload)

