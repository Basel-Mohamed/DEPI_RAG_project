from __future__ import annotations

from typing import Any

from app.services.reranking.base_reranker import BaseRerankerService, RerankerServiceError


class CohereRerankerService(BaseRerankerService):
    """Cohere-based reranker service powered by LangChain's Cohere wrapper."""

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = "rerank-v3.5",
        top_n: int | None = None,
    ) -> None:
        super().__init__(top_n=top_n)
        self.api_key = api_key
        self.model_name = model_name
        self._reranker: Any | None = None

    def _get_reranker(self) -> Any:
        """Lazily create and reuse this service's LangChain Cohere reranker."""

        if self._reranker is not None:
            return self._reranker

        try:
            from langchain_cohere import CohereRerank
        except ImportError as exc:
            raise RerankerServiceError(
                "The 'langchain-cohere' package is required to use CohereRerankerService."
            ) from exc

        self._reranker = CohereRerank(
            cohere_api_key=self.api_key,
            model=self.model_name,
            top_n=self.top_n,
        )
        return self._reranker
