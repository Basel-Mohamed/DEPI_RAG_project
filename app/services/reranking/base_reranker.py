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

    def score(self, query: str, documents: list[RetrievedContext]) -> list[float]:
        """Return relevance scores aligned with the original document order."""

        if not query.strip() or not documents:
            return []

        document_texts = [document.content for document in documents]

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
        documents: list[RetrievedContext],
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
            self._with_rerank_score(document, score=float(scores[index]))
            for index, document in ranked_pairs
        ]

        if top_k is None:
            return (
                normalized_documents[: self.top_n]
                if self.top_n is not None
                else normalized_documents
            )
        return normalized_documents[:top_k]

    def _with_rerank_score(
        self,
        document: RetrievedContext,
        *,
        score: float | None = None,
    ) -> RetrievedContext:
        """Return a copy of the context with rerank score metadata."""

        metadata = dict(document.metadata or {})
        if score is not None:
            metadata = {**metadata, "rerank_score": score}
        return RetrievedContext(
            id=document.id,
            title=document.title,
            content=document.content,
            metadata=metadata,
        )
