from __future__ import annotations

import json
from typing import Any

from app.services.reranker.base_reranker import BaseRerankerService, RerankerServiceError


class _AzureCohereRerankerAdapter:
    """Call the Azure Cohere rerank endpoint directly and normalize responses."""

    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name

    def rerank(
        self,
        *,
        documents: list[str],
        query: str,
        top_n: int,
    ) -> list[dict[str, float | int]]:
        try:
            import httpx
        except ImportError as exc:
            raise RerankerServiceError(
                "The 'httpx' package is required to use AzureCohereRerankerService."
            ) from exc

        response = httpx.post(
            self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "api-key": self._api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": self._model_name,
                "documents": documents,
                "query": query,
                "top_n": top_n,
            },
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise RerankerServiceError(
                f"Azure Cohere rerank request failed with status {response.status_code}: {response.text}"
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RerankerServiceError(
                f"Azure Cohere rerank returned a non-JSON response: {response.text}"
            ) from exc

        results = payload.get("results", [])
        return [
            {
                "index": int(result.index),
                "relevance_score": float(result.relevance_score),
            }
            for result in [
                type(
                    "AzureRerankResult",
                    (),
                    {
                        "index": item.get("index", 0),
                        "relevance_score": item.get("relevance_score", 0.0),
                    },
                )()
                for item in results
            ]
        ]

    def compress_documents(self, *, documents: list[Any], query: str) -> list[Any]:
        raise RerankerServiceError(
            "Azure Cohere reranker compression is not implemented for the current provider endpoint."
        )


class AzureCohereRerankerService(BaseRerankerService):
    """Azure Cohere-based reranker service using Cohere's Azure base URL."""

    _singleton_reranker: Any | None = None
    _singleton_config: tuple[str, str, str, int | None] | None = None

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        model_name: str = "model",
        top_n: int | None = None,
    ) -> None:
        super().__init__(top_n=top_n)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def _get_reranker(self) -> Any:
        """Lazily create and reuse a process-wide Azure Cohere reranker."""

        config = (self.api_key, self.base_url, self.model_name, self.top_n)

        if self.__class__._singleton_reranker is not None:
            if self.__class__._singleton_config != config:
                raise RerankerServiceError(
                    "AzureCohereRerankerService singleton was already initialized with a different configuration."
                )
            return self.__class__._singleton_reranker

        self.__class__._singleton_reranker = _AzureCohereRerankerAdapter(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
        )
        self.__class__._singleton_config = config
        return self.__class__._singleton_reranker
