from app.models.document import Document


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> Document:
        self._documents[document.id] = document
        return document

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def list_all(self) -> list[Document]:
        return list(self._documents.values())

