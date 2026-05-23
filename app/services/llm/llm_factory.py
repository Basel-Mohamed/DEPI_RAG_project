from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.services.llm.providers.azure_llm import AzureLlmService
from app.services.llm.providers.base_llm import BaseLlmService, LlmServiceError
from app.services.llm.providers.cohere_llm import CohereLlmService

_service_singleton: BaseLlmService | None = None
_service_config: tuple[Any, ...] | None = None
logger = logging.getLogger(__name__)


def create_llm_service(settings: Settings) -> BaseLlmService:
    """Create the configured LLM provider implementation for the pipeline."""

    provider = settings.llm_provider.lower()
    config = _build_service_config(settings, provider)

    global _service_singleton, _service_config
    if _service_singleton is not None:
        if _service_config != config:
            logger.error("LLM service singleton requested with conflicting provider configuration")
            raise LlmServiceError(
                "LLM service singleton was already initialized with a different configuration."
            )
        return _service_singleton

    if provider == "azure":
        _validate_azure_settings(settings)
        logger.info("initializing Azure LLM service deployment=%s", settings.azure_openai_chat_deployment)
        _service_singleton = AzureLlmService(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            deployment_name=settings.azure_openai_chat_deployment,
            api_version=settings.azure_openai_api_version,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        _service_config = config
        return _service_singleton

    if provider == "cohere":
        _validate_cohere_settings(settings)
        logger.info("initializing Cohere LLM service model=%s", settings.cohere_chat_model)
        _service_singleton = CohereLlmService(
            api_key=settings.cohere_api_key,
            model_name=settings.cohere_chat_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        _service_config = config
        return _service_singleton

    logger.error("unsupported LLM provider requested provider=%s", provider)
    raise LlmServiceError(
        "Unsupported LLM provider. Set LLM_PROVIDER to 'azure' or 'cohere'."
    )


def _build_service_config(settings: Settings, provider: str) -> tuple[Any, ...]:
    if provider == "azure":
        return (
            provider,
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_chat_deployment,
            settings.azure_openai_api_version,
            settings.llm_temperature,
            settings.llm_max_tokens,
        )

    if provider == "cohere":
        return (
            provider,
            settings.cohere_api_key,
            settings.cohere_chat_model,
            settings.llm_temperature,
            settings.llm_max_tokens,
        )

    return (provider,)


def _validate_azure_settings(settings: Settings) -> None:
    missing_fields: list[str] = []
    if not settings.azure_openai_endpoint:
        missing_fields.append("AZURE_OPENAI_ENDPOINT")
    if not settings.azure_openai_api_key:
        missing_fields.append("AZURE_OPENAI_API_KEY")
    if not settings.azure_openai_chat_deployment:
        missing_fields.append("AZURE_OPENAI_CHAT_DEPLOYMENT")

    if missing_fields:
        logger.error("Azure LLM configuration incomplete missing=%s", ", ".join(missing_fields))
        raise LlmServiceError(
            "Azure LLM configuration is incomplete. Missing: "
            + ", ".join(missing_fields)
        )


def _validate_cohere_settings(settings: Settings) -> None:
    missing_fields: list[str] = []
    if not settings.cohere_api_key:
        missing_fields.append("COHERE_API_KEY")

    if missing_fields:
        logger.error("Cohere LLM configuration incomplete missing=%s", ", ".join(missing_fields))
        raise LlmServiceError(
            "Cohere LLM configuration is incomplete. Missing: "
            + ", ".join(missing_fields)
        )
