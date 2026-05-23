from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.services.types import RetrievedContext


class RerankerServiceError(RuntimeError):
    """Raised when the reranker cannot score or reorder documents."""


class BaseRerankerService(ABC):
    """Abstract base class for reranker services."""

    def __init__(self, top_n: int | None = None) -> None:
        self.top_n = top_n

    @abstractmethod
    def _get_reranker(self) -> Any:
        """Get the underlying reranker implementation."""
        pass

    def _extract_content_and_metadata(self, document: Any) -> tuple[str, dict[str, Any]]:
        """Extract content and metadata from document in a consistent way."""
        if isinstance(document, dict):
            content = document.get('content', '')
            metadata = {
                **(document.get('metadata', {}) or {}),
                'id': document.get('id', ''),
                'title': document.get('title', ''),
            }
        elif isinstance(document, RetrievedContext):
            content = document.content
            metadata = {
                **(document.metadata or {}),
                'id': document.id,
                'title': document.title,
            }
        elif hasattr(document, 'page_content'):
            content = document.page_content
            metadata = getattr(document, 'metadata', {})
        else:
            content = str(getattr(document, 'content', '') or str(document))
            metadata = {
                **(getattr(document, 'metadata', {}) or {}),
                'id': getattr(document, 'id', ''),
                'title': getattr(document, 'title', ''),
            }
        if not isinstance(metadata, dict):
            metadata = {'raw_metadata': metadata}
        return content, metadata

    def score(self, query: str, documents: list[Any]) -> list[float]:
        """Return relevance scores aligned with the original document order."""

        if not query.strip() or not documents:
            return []

        document_texts = [
            self._extract_content_and_metadata(document)[0]
            for document in documents
        ]

        try:
            results = self._get_reranker().rerank(
                documents=document_texts,
                query=query.strip(),
                top_n=len(document_texts),
            )
        except Exception as exc:
            raise RerankerServiceError("Reranking failed.") from exc

        scores = [0.0] * len(document_texts)
        for result in results:
            scores[int(result["index"])] = float(result["relevance_score"])
        return scores

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Return the documents ordered by relevance score."""

        if not query.strip() or not documents:
            return []

        if top_k is not None and top_k <= 0:
            return []

        scores = self.score(query, documents)
        indexed_documents = list(enumerate(documents))
        ranked_pairs = sorted(
            indexed_documents,
            key=lambda item: scores[item[0]],
            reverse=True,
        )
        normalized_documents = [
            self._convert_to_retrieved_context(document, score=float(scores[index]))
            for index, document in ranked_pairs
        ]

        if top_k is None:
            return (
                normalized_documents[: self.top_n]
                if self.top_n is not None
                else normalized_documents
            )
        return normalized_documents[:top_k]

    def _convert_to_retrieved_context(
        self,
        document: Any,
        *,
        score: float | None = None,
    ) -> RetrievedContext:
        """Convert document to RetrievedContext for output consistency."""
        content, metadata = self._extract_content_and_metadata(document)
        if score is not None:
            metadata = {**metadata, "rerank_score": score}
        return RetrievedContext(
            id=str(metadata.get('_id', metadata.get('id', metadata.get('source_id', '')))),
            title=str(metadata.get('title', '')),
            content=content,
            metadata=metadata,
        )
