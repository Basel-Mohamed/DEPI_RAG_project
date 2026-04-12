from app.repositories.document_repository import InMemoryDocumentRepository
from app.repositories.vector_store import InMemoryVectorStore
from app.schemas.query import QueryRequest, QueryResponse


class RAGService:
    def __init__(
        self,
        document_repository: InMemoryDocumentRepository,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self.document_repository = document_repository
        self.vector_store = vector_store

    def answer_question(self, payload: QueryRequest) -> QueryResponse:
        document_ids = self.vector_store.similarity_search(
            query=payload.question,
            top_k=payload.top_k,
        )
        documents = [
            document
            for document_id in document_ids
            if (document := self.document_repository.get(document_id)) is not None
        ]

        if not documents:
            return QueryResponse(
                answer="I could not find relevant context in the current document store.",
                sources=[],
            )

        context = "\n\n".join(
            f"{document.title}: {document.content}" for document in documents
        )
        answer = (
            "Stubbed RAG answer based on retrieved context.\n\n"
            f"Question: {payload.question}\n\n"
            f"Context:\n{context}"
        )
        return QueryResponse(
            answer=answer,
            sources=[document.id for document in documents],
        )
