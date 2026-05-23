from __future__ import annotations

import os
from typing import Any

import pytest

from app.core.config import Settings
from app.services.llm.llm_factory import create_llm_service
from app.services.rag.rag_inference import RagInferencePipeline


class FixedVectorStore:
    def search(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "integration-policy-1",
                "text": (
                    "The support policy says warranty claims must be submitted "
                    "within 14 days. Customers must include receipt code ALPHA-7 "
                    "when opening the claim."
                ),
                "score": 0.99,
                "metadata": {
                    "source": "integration-policy.pdf",
                    "page_number": 1,
                },
            }
        ]


def _real_llm_settings() -> Settings:
    settings = Settings(
        RERANKED_CONTEXT_TOP_K=1,
        RETRIEVAL_CANDIDATE_TOP_K=1,
        reranker_provider=None,
        llm_temperature=0.0,
        llm_max_tokens=120,
    )

    provider = settings.llm_provider.lower()
    if provider == "azure":
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
                "AZURE_OPENAI_CHAT_DEPLOYMENT": settings.azure_openai_chat_deployment,
            }.items()
            if not value
        ]
    elif provider == "cohere":
        missing = ["COHERE_API_KEY"] if not settings.cohere_api_key else []
    else:
        missing = [f"unsupported provider: {settings.llm_provider}"]

    if missing:
        pytest.skip("Missing real LLM configuration: " + ", ".join(missing))

    return settings


@pytest.mark.integration
def test_rag_pipeline_calls_real_llm_and_returns_grounded_answer() -> None:
    if os.getenv("RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("Set RUN_REAL_LLM_TESTS=1 to call the configured LLM API.")

    settings = _real_llm_settings()
    pipeline = RagInferencePipeline(
        vector_store=FixedVectorStore(),
        llm_service=create_llm_service(settings),
        settings=settings,
    )

    try:
        response = pipeline.run(
            "According to the support policy, when must warranty claims be submitted "
            "and which receipt code is required?"
        )
    except Exception:
        pytest.fail(
            "The real LLM request failed. Check network access and LLM credentials.",
            pytrace=False,
        )

    answer = response["answer"]
    assert answer.strip()
    assert "14" in answer
    assert "ALPHA-7" in answer
    assert response["sources"][0]["id"] == "integration-policy-1"
    assert response["retrieval"]["documents"] == 1

