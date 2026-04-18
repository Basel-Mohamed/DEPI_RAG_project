from __future__ import annotations

from typing import Any

from app.services.reranker.base_reranker import BaseRerankerService, RerankerServiceError


class CohereRerankerService(BaseRerankerService):
    """Cohere-based reranker service powered by LangChain's Cohere wrapper."""

    _singleton_reranker: Any | None = None
    _singleton_config: tuple[str, str, int | None] | None = None

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

    def _get_reranker(self) -> Any:
        """Lazily create and reuse a process-wide LangChain Cohere reranker."""

        config = (self.api_key, self.model_name, self.top_n)

        if self.__class__._singleton_reranker is not None:
            if self.__class__._singleton_config != config:
                raise RerankerServiceError(
                    "CohereRerankerService singleton was already initialized with a different configuration."
                )
            return self.__class__._singleton_reranker

        try:
            from langchain_cohere import CohereRerank
        except ImportError as exc:
            raise RerankerServiceError(
                "The 'langchain-cohere' package is required to use CohereRerankerService."
            ) from exc

        self.__class__._singleton_reranker = CohereRerank(
            cohere_api_key=self.api_key,
            model=self.model_name,
            top_n=self.top_n,
        )
        self.__class__._singleton_config = config
        return self.__class__._singleton_reranker
