from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from app.services.reranking.base_reranker import BaseRerankerService, RerankerServiceError
from app.services.reranking.providers.azure_reranker import AzureCohereRerankerService
from app.services.reranking.providers.cohere_reranker import CohereRerankerService

if TYPE_CHECKING:
    from app.core.config import Settings


class RerankerType(Enum):
    """Supported reranker types."""

    COHERE = "cohere"
    AZURE_COHERE = "azure_cohere"


class RerankerFactory:
    """Factory class for creating reranker service instances."""

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

        if reranker_type == RerankerType.COHERE:
            return CohereRerankerService(
                api_key=kwargs["api_key"],
                model_name=kwargs.get("model_name", "rerank-v3.5"),
                top_n=kwargs.get("top_n"),
            )
        if reranker_type == RerankerType.AZURE_COHERE:
            return AzureCohereRerankerService(
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                model_name=kwargs.get("model_name", "model"),
                top_n=kwargs.get("top_n"),
            )

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


def create_reranker_service(settings: Settings) -> BaseRerankerService | None:
    """Create the configured reranker service from application settings.

    This mirrors the LLM factory style: callers provide ``Settings`` and the
    reranker layer owns provider selection, credential validation, and model
    defaults. Returning ``None`` means reranking is disabled.
    """

    provider = (getattr(settings, "reranker_provider", None) or "").lower()
    if provider in {"", "none", "off", "false", "disabled"}:
        return None

    if provider == RerankerType.COHERE.value:
        api_key = getattr(settings, "cohere_api_key", None)
        if not api_key:
            raise RerankerServiceError(
                "Cohere reranker configuration is incomplete. Missing: COHERE_API_KEY"
            )
        return RerankerFactory.create_cohere_reranker(
            api_key=api_key,
            model_name=getattr(settings, "cohere_rerank_model", "rerank-v3.5"),
        )

    if provider == RerankerType.AZURE_COHERE.value:
        api_key = getattr(settings, "azure_cohere_api_key", None)
        base_url = getattr(settings, "azure_cohere_base_url", None)
        if not api_key or not base_url:
            missing_fields = []
            if not api_key:
                missing_fields.append("AZURE_COHERE_API_KEY")
            if not base_url:
                missing_fields.append("AZURE_COHERE_BASE_URL")
            raise RerankerServiceError(
                "Azure Cohere reranker configuration is incomplete. Missing: "
                + ", ".join(missing_fields)
            )
        return RerankerFactory.create_azure_cohere_reranker(
            api_key=api_key,
            base_url=base_url,
            model_name=getattr(settings, "azure_cohere_model", "model"),
        )

    raise RerankerServiceError(
        "Unsupported reranker provider. Set RERANKER_PROVIDER to "
        "'cohere', 'azure_cohere', or 'none'."
    )
