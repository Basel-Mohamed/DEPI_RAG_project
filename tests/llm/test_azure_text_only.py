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


def _legacy_non_text_documents() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            id="chunk-1",
            title="Policy p.1",
            content="The text says receipt code ALPHA-7.",
            metadata={
                "source": "policy.pdf",
                "page_number": 1,
                "page_image_url": "/media/page-images/policy/1.png",
                "page_image_mime_type": "image/png",
            },
        )
    ]


def test_azure_build_messages_are_text_only_and_strip_non_text_metadata() -> None:
    service = _service()

    messages = service.build_messages(
        "What receipt code appears on the page?",
        _legacy_non_text_documents(),
    )

    assert isinstance(messages[1]["content"], str)
    assert "ALPHA-7" in messages[1]["content"]
    assert "page_image_url" not in messages[1]["content"]
    assert "image_url" not in messages[1]["content"]
