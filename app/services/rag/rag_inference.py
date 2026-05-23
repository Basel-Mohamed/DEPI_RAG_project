from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, settings as global_settings
from app.services.llm.providers.base_llm import (
    DEFAULT_FALLBACK_ANSWER,
    BaseLlmService,
)
from app.services.rag.inference_helpers.response import RagResponseBuilder
from app.services.rag.inference_helpers.retrieval import result_to_context
from app.services.types import RetrievedContext

if TYPE_CHECKING:
    from app.services.reranking.base_reranker import BaseRerankerService
    from app.services.vectorstore.qdrant_store import QdrantService, SearchMode

logger = logging.getLogger(__name__)

GeneratedPayload = dict[str, Any]


class RagInferencePipeline:
    """Orchestrate retrieval, optional reranking, and LLM generation.

    The heavy formatting details live in ``RagResponseBuilder`` so the pipeline
    can focus on orchestration.
    """

    def __init__(
        self,
        *,
        vector_store: QdrantService,
        llm_service: BaseLlmService | None = None,
        reranker: BaseRerankerService | None = None,
        settings: Settings | None = None,
        response_builder: RagResponseBuilder | None = None,
    ) -> None:
        """Create the pipeline from injected infrastructure dependencies."""

        self.settings = settings or global_settings

        if llm_service is None:
            from app.services.llm.llm_factory import create_llm_service

            llm_service = create_llm_service(self.settings)

        self.vector_store = vector_store
        self.llm_service = llm_service
        if reranker is None:
            from app.services.reranking.reranker_factory import create_reranker_service

            reranker = create_reranker_service(self.settings)

        self.reranker = reranker
        self.response_builder = response_builder or RagResponseBuilder()

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: SearchMode | str | None = None,
        score_threshold: float | None = None,
        include_sources: bool = True,
    ) -> GeneratedPayload:
        """Return a full non-streaming RAG response for ``question``."""

        context = self._get_documents_or_fallback(
            question,
            top_k=top_k,
            mode=mode,
            score_threshold=score_threshold,
        )
        if context is None:
            return self.response_builder.empty_response(DEFAULT_FALLBACK_ANSWER)

        clean_question, documents = context

        raw_answer = self.llm_service.generate(clean_question, documents)

        return self.response_builder.build(
            raw_answer,
            documents,
            include_sources=include_sources,
            retrieval=self._retrieval_metadata(documents, mode),
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: SearchMode | str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve and rerank context documents for the question."""

        clean_question = question.strip()
        if not clean_question:
            return []

        final_top_k = int(top_k or self.settings.RAG_RERANK_TOP_K)
        search_top_k = int(self.settings.RAG_RETRIEVAL_TOP_K)

        raw_results = self.vector_store.search(
            query=clean_question,
            top_k=search_top_k,
            mode=mode,
            score_threshold=score_threshold,
        )

        retrieved_documents = [result_to_context(result) for result in raw_results]
        ranked_documents = self._rerank(
            clean_question,
            retrieved_documents,
            top_k=final_top_k,
        )

        return ranked_documents

    def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: SearchMode | str | None = None,
        score_threshold: float | None = None,
        include_sources: bool = True,
    ) -> Iterator[GeneratedPayload]:
        """Yield normalized answer deltas followed by source metadata."""

        context = self._get_documents_or_fallback(
            question,
            top_k=top_k,
            mode=mode,
            score_threshold=score_threshold,
        )
        if context is None:
            yield self.response_builder.empty_response(DEFAULT_FALLBACK_ANSWER)
            return

        clean_question, documents = context

        for raw_chunk in self.llm_service.stream(clean_question, documents):
            yield self.response_builder.build_delta(raw_chunk)

        yield {
            "event": "sources",
            "answer": "",
            "content": [],
            "sources": (
                self.response_builder.build_sources(documents)
                if include_sources
                else []
            ),
            "retrieval": self._retrieval_metadata(documents, mode),
        }

    def _get_documents_or_fallback(
        self,
        question: str,
        *,
        top_k: int | None,
        mode: SearchMode | str | None,
        score_threshold: float | None,
    ) -> tuple[str, list[RetrievedContext]] | None:
        """Return a clean question and contexts, or ``None`` for fallback answers."""

        clean_question = question.strip()
        if not clean_question:
            return None

        documents = self.retrieve(
            clean_question,
            top_k=top_k,
            mode=mode,
            score_threshold=score_threshold,
        )
        if not documents:
            return None

        return clean_question, documents

    def _rerank(
        self,
        question: str,
        documents: list[RetrievedContext],
        *,
        top_k: int,
    ) -> list[RetrievedContext]:
        """Rerank documents, falling back on retrieval order."""

        if not documents or top_k <= 0:
            return []
        if self.reranker is None:
            return documents[:top_k]

        try:
            return self.reranker.rerank(question, documents, top_k=top_k)
        except Exception:
            logger.exception("Reranking failed; falling back to retrieval order.")
            return documents[:top_k]

    def _retrieval_metadata(
        self,
        documents: list[RetrievedContext],
        mode: SearchMode | str | None,
    ) -> dict[str, Any]:
        """Return retrieval metadata included in responses."""

        return {
            "documents": len(documents),
            "mode": str(mode or getattr(self.settings, "RETRIEVAL_MODE", "hybrid")),
        }
