from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_service import RAGService


class QueryController:
    def __init__(self, rag_service: RAGService) -> None:
        self.rag_service = rag_service

    def query(self, payload: QueryRequest) -> QueryResponse:
        return self.rag_service.answer_question(payload)

