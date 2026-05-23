from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, settings as global_settings
from app.controllers.monitoring_controller import MonitoringMetrics
from app.services.llm.providers.base_llm import (
    DEFAULT_FALLBACK_ANSWER,
    BaseLlmService,
)
from app.services.rag.inference_helpers.response import RagResponseBuilder
from app.services.rag.inference_helpers.retrieval import RetrievalPolicy
from app.services.types import RetrievedContext

if TYPE_CHECKING:
    from app.services.reranking.base_reranker import BaseRerankerService
    from app.services.vectorstore.qdrant_store import QdrantService, SearchMode

logger = logging.getLogger(__name__)

GeneratedPayload = dict[str, Any]


class RagInferencePipeline:
    """Orchestrate retrieval, optional reranking, and LLM generation.

    The heavy formatting details live in helper classes:
    ``RetrievalPolicy`` handles result conversion and text-context rules, and
    ``RagResponseBuilder`` handles source payloads and LLM output normalization.
    """

    def __init__(
        self,
        *,
        vector_store: QdrantService | None = None,
        llm_service: BaseLlmService | None = None,
        reranker: BaseRerankerService | None = None,
        settings: Settings | None = None,
        retrieval_policy: RetrievalPolicy | None = None,
        response_builder: RagResponseBuilder | None = None,
    ) -> None:
        """Create the pipeline and lazily construct missing dependencies."""

        self.settings = settings or global_settings

        if vector_store is None:
            from app.services.embedding.embedding_service import EmbeddingService
            from app.services.vectorstore.qdrant_store import QdrantService

            embedding_service = EmbeddingService(settings=self.settings)
            vector_store = QdrantService(
                embedding_service=embedding_service,
                settings=self.settings,
            )

        if llm_service is None:
            from app.services.llm.llm_factory import create_llm_service

            llm_service = create_llm_service(self.settings)

        self.vector_store = vector_store
        self.llm_service = llm_service
        if reranker is None:
            from app.services.reranking.reranker_factory import create_reranker_service

            reranker = create_reranker_service(self.settings)

        self.reranker = reranker
        self.retrieval_policy = retrieval_policy or RetrievalPolicy()
        self.response_builder = response_builder or RagResponseBuilder()

    def run(
        self,
        question: str,
        *,
        mode: SearchMode | str | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
        include_sources: bool = True,
    ) -> GeneratedPayload:
        """Return a full non-streaming RAG response for ``question``."""

        clean_question = question.strip()
        if not clean_question:
            return self.response_builder.empty_response(DEFAULT_FALLBACK_ANSWER)

        documents = self.retrieve(
            clean_question,
            mode=mode,
            filter_field=filter_field,
            filter_value=filter_value,
            score_threshold=score_threshold,
        )
        if not documents:
            return self.response_builder.empty_response(DEFAULT_FALLBACK_ANSWER)

        start = time.perf_counter()
        raw_answer = self.llm_service.generate(clean_question, documents)
        MonitoringMetrics.record_llm(
            (time.perf_counter() - start) * 1000,
            self._estimate_tokens(str(raw_answer)),
        )

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
        mode: SearchMode | str | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve and rerank context documents for the question."""

        clean_question = question.strip()
        if not clean_question:
            return []

        final_top_k = int(self.settings.RAG_RERANK_TOP_K)
        search_top_k = int(self.settings.RAG_RETRIEVAL_TOP_K)

        raw_results = self.vector_store.search(
            query=clean_question,
            top_k=search_top_k,
            mode=mode,
            filter_field=filter_field,
            filter_value=filter_value,
            score_threshold=self.retrieval_policy.effective_score_threshold(
                clean_question,
                score_threshold,
            ),
        )

        retrieved_documents = [
            self.retrieval_policy.result_to_context(result)
            for result in raw_results
        ]
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
        mode: SearchMode | str | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
        include_sources: bool = True,
    ) -> Iterator[GeneratedPayload]:
        """Yield normalized answer deltas followed by source metadata."""

        clean_question = question.strip()
        if not clean_question:
            yield self.response_builder.empty_response(DEFAULT_FALLBACK_ANSWER)
            return

        documents = self.retrieve(
            clean_question,
            mode=mode,
            filter_field=filter_field,
            filter_value=filter_value,
            score_threshold=score_threshold,
        )
        if not documents:
            yield self.response_builder.empty_response(DEFAULT_FALLBACK_ANSWER)
            return

        start = time.perf_counter()
        token_count = 0
        try:
            for raw_chunk in self.llm_service.stream(clean_question, documents):
                token_count += self._estimate_tokens(str(raw_chunk))
                yield self.response_builder.build_delta(raw_chunk)
        finally:
            MonitoringMetrics.record_llm(
                (time.perf_counter() - start) * 1000,
                token_count,
            )

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

    def close(self) -> None:
        """Release resources held by the vector store."""

        close = getattr(self.vector_store, "close", None)
        if callable(close):
            close()

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
            start = time.perf_counter()
            ranked_documents = self.reranker.rerank(question, documents, top_k=top_k)
            MonitoringMetrics.record_reranking_latency(
                (time.perf_counter() - start) * 1000
            )
            return ranked_documents
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
