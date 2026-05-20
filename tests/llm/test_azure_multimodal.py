from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.llm.providers.azure_llm import AzureLlmService
from app.services.types import RetrievedContext


class FakeAzureChatModel:
    def __init__(self) -> None:
        self.invoked_with: list[Any] | None = None
        self.streamed_with: list[Any] | None = None

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        self.invoked_with = messages
        return SimpleNamespace(content="The image and text were received.")

    def stream(self, messages: list[Any]):
        self.streamed_with = messages
        yield SimpleNamespace(content="The image ")
        yield SimpleNamespace(content="was received.")


def _service_with_fake_model(fake_model: FakeAzureChatModel) -> AzureLlmService:
    service = AzureLlmService(
        azure_endpoint="https://example.openai.azure.com",
        api_key="test-key",
        deployment_name="gpt-4o",
        api_version="2024-02-01",
        temperature=0.0,
        max_tokens=100,
    )
    service._get_llm = lambda: fake_model  # type: ignore[method-assign]
    return service


def _image_documents() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            id="chunk-1",
            title="Policy p.1",
            content="The page image shows receipt code ALPHA-7.",
            metadata={
                "source": "policy.pdf",
                "page_number": 1,
                "page_image_base64": "data:image/png;base64,abc123",
                "page_image_mime_type": "image/png",
            },
        )
    ]


def test_azure_generate_sends_retrieved_images_as_multimodal_blocks() -> None:
    fake_model = FakeAzureChatModel()
    service = _service_with_fake_model(fake_model)

    answer = service.generate(
        "What receipt code appears on the page?",
        _image_documents(),
    )

    assert answer == "The image and text were received."
    assert fake_model.invoked_with is not None
    human_message = fake_model.invoked_with[1]
    assert isinstance(human_message.content, list)
    assert human_message.content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }
    assert "abc123" not in human_message.content[0]["text"]
    assert "page_image_base64" not in human_message.content[0]["text"]


def test_azure_stream_sends_retrieved_images_as_multimodal_blocks() -> None:
    fake_model = FakeAzureChatModel()
    service = _service_with_fake_model(fake_model)

    chunks = list(
        service.stream(
            "What receipt code appears on the page?",
            _image_documents(),
        )
    )

    assert "".join(chunks) == "The image was received."
    assert fake_model.streamed_with is not None
    human_message = fake_model.streamed_with[1]
    assert human_message.content[1]["image_url"]["url"] == "data:image/png;base64,abc123"
