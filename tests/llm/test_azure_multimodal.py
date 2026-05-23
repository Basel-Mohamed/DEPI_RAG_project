from __future__ import annotations

from app.services.llm.providers.azure_llm import AzureLlmService
from app.services.types import RetrievedContext


def _service() -> AzureLlmService:
    return AzureLlmService(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        deployment_name="gpt-4o",
        api_version="2024-02-01",
        temperature=0.0,
        max_tokens=100,
    )


def _documents_with_image_metadata() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            id="chunk-1",
            title="Policy p.1",
            content="Receipt code ALPHA-7 is required.",
            metadata={
                "source": "policy.pdf",
                "page_number": 1,
                "page_image_base64": "data:image/png;base64,abc123",
                "page_image_mime_type": "image/png",
            },
        )
    ]


def test_azure_build_messages_ignores_retrieved_image_metadata() -> None:
    messages = _service().build_messages(
        "What receipt code appears on the page?",
        _documents_with_image_metadata(),
    )

    assert isinstance(messages[1]["content"], str)
    assert "Receipt code ALPHA-7" in messages[1]["content"]
    assert "abc123" not in messages[1]["content"]
    assert "page_image_base64" not in messages[1]["content"]
