from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.reranking.base_reranker import RerankerServiceError
from app.services.reranking.reranker_factory import create_reranker_service


def test_create_reranker_service_returns_none_when_disabled() -> None:
    service = create_reranker_service(SimpleNamespace(reranker_provider="none"))

    assert service is None


def test_create_reranker_service_fails_when_cohere_key_missing() -> None:
    settings = SimpleNamespace(reranker_provider="cohere", cohere_api_key=None)

    with pytest.raises(RerankerServiceError, match="COHERE_API_KEY"):
        create_reranker_service(settings)


def test_create_reranker_service_fails_when_azure_cohere_config_missing() -> None:
    settings = SimpleNamespace(
        reranker_provider="azure_cohere",
        azure_cohere_api_key=None,
        azure_cohere_base_url=None,
    )

    with pytest.raises(RerankerServiceError, match="AZURE_COHERE_API_KEY"):
        create_reranker_service(settings)
