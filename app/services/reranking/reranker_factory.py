from __future__ import annotations

from enum import Enum
from typing import Any

from app.services.reranker.base_reranker import BaseRerankerService
from app.services.reranker.providers.azure_reranker import AzureCohereRerankerService
from app.services.reranker.providers.cohere_reranker import CohereRerankerService


class RerankerType(Enum):
    """Supported reranker types."""
    COHERE = "cohere"
    AZURE_COHERE = "azure_cohere"


class RerankerFactory:
    """Factory class for creating reranker service instances."""

    _singleton_service: BaseRerankerService | None = None
    _singleton_config: tuple[Any, ...] | None = None

    @staticmethod
    def create_reranker(
        reranker_type: RerankerType | str,
        **kwargs: Any,
    ) -> BaseRerankerService:
        """Create a reranker service instance based on the specified type.

        Args:
            reranker_type: The type of reranker to create
            **kwargs: Configuration parameters specific to the reranker type

        Returns:
            A configured reranker service instance

        Raises:
            ValueError: If an unsupported reranker type is specified
        """
        if isinstance(reranker_type, str):
            try:
                reranker_type = RerankerType(reranker_type.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported reranker type: {reranker_type}. "
                    f"Supported types: {[rt.value for rt in RerankerType]}"
                )

        config = RerankerFactory._build_config(reranker_type, **kwargs)
        if RerankerFactory._singleton_service is not None:
            if RerankerFactory._singleton_config != config:
                raise ValueError(
                    "Reranker singleton was already initialized with a different configuration."
                )
            return RerankerFactory._singleton_service

        if reranker_type == RerankerType.COHERE:
            RerankerFactory._singleton_service = CohereRerankerService(
                api_key=kwargs["api_key"],
                model_name=kwargs.get("model_name", "rerank-v3.5"),
                top_n=kwargs.get("top_n"),
            )
            RerankerFactory._singleton_config = config
            return RerankerFactory._singleton_service
        elif reranker_type == RerankerType.AZURE_COHERE:
            RerankerFactory._singleton_service = AzureCohereRerankerService(
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                model_name=kwargs.get("model_name", "model"),
                top_n=kwargs.get("top_n"),
            )
            RerankerFactory._singleton_config = config
            return RerankerFactory._singleton_service
        else:
            raise ValueError(
                f"Unsupported reranker type: {reranker_type}. "
                f"Supported types: {[rt.value for rt in RerankerType]}"
            )

    @staticmethod
    def create_cohere_reranker(
        api_key: str,
        *,
        model_name: str = "rerank-v3.5",
        top_n: int | None = None,
    ) -> CohereRerankerService:
        """Create a Cohere reranker service instance.

        Args:
            api_key: Cohere API key
            model_name: Model name to use
            top_n: Maximum number of results to return

        Returns:
            A configured Cohere reranker service instance
        """
        return RerankerFactory.create_reranker(
            RerankerType.COHERE,
            api_key=api_key,
            model_name=model_name,
            top_n=top_n,
        )

    @staticmethod
    def create_azure_cohere_reranker(
        api_key: str,
        base_url: str,
        *,
        model_name: str = "model",
        top_n: int | None = None,
    ) -> AzureCohereRerankerService:
        """Create an Azure Cohere reranker service instance.

        Args:
            api_key: Azure API key for the Cohere rerank deployment
            base_url: Azure Cohere rerank base URL
            model_name: Model name to use
            top_n: Maximum number of results to return

        Returns:
            A configured Azure Cohere reranker service instance
        """
        return RerankerFactory.create_reranker(
            RerankerType.AZURE_COHERE,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            top_n=top_n,
        )

    @staticmethod
    def _build_config(
        reranker_type: RerankerType,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if reranker_type == RerankerType.COHERE:
            return (
                reranker_type.value,
                kwargs["api_key"],
                kwargs.get("model_name", "rerank-v3.5"),
                kwargs.get("top_n"),
            )

        if reranker_type == RerankerType.AZURE_COHERE:
            return (
                reranker_type.value,
                kwargs["api_key"],
                kwargs["base_url"],
                kwargs.get("model_name", "model"),
                kwargs.get("top_n"),
            )

        return (reranker_type.value,)
