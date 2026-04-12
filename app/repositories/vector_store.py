from app.models.document import Document


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._index: dict[str, str] = {}

    def add_document(self, document: Document) -> None:
        self._index[document.id] = document.content

    def similarity_search(self, query: str, top_k: int = 3) -> list[str]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        if not query_terms:
            return []

        scored_documents: list[tuple[str, int]] = []
        for document_id, content in self._index.items():
            content_terms = {term.lower() for term in content.split() if term.strip()}
            score = len(query_terms.intersection(content_terms))
            if score > 0:
                scored_documents.append((document_id, score))

        scored_documents.sort(key=lambda item: item[1], reverse=True)
        return [document_id for document_id, _ in scored_documents[:top_k]]

