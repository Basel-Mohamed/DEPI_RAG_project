from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.media import MediaExtractor
from app.services.rag.inference_helpers.response import RagResponseBuilder
from app.services.rag.rag_inference import RagInferencePipeline
from app.services.types import RetrievedContext


class FakeVectorStore:
    def search(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "chunk-1",
                "text": "Refunds are available for eligible orders within 30 days.",
                "score": 0.72,
                "metadata": {
                    "source": "policy.pdf",
                    "page_number": 2,
                    "image_url": "https://example.test/refund-flow.png",
                },
            },
            {
                "id": "chunk-2",
                "text": "Customers need the order id to start the refund workflow.",
                "score": 0.68,
                "metadata": {"source": "policy.pdf", "page_number": 3},
            },
        ]


class FakeVisualVectorStore:
    def __init__(self) -> None:
        self.last_score_threshold: float | None = None

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_score_threshold = kwargs.get("score_threshold")
        return [
            {
                "id": "docx-image-placeholder",
                "text": "## 2 جدول التدريب <!-- image -->",
                "score": 1.0,
                "metadata": {
                    "source": "depi_test.docx",
                    "page_number": 1,
                },
            },
            {
                "id": "pdf-page-image",
                "text": "جدول التدريب",
                "score": 0.3333,
                "metadata": {
                    "source": "depi_test.pdf",
                    "page_number": 3,
                    "page_image_base64": "data:image/png;base64,schedule",
                    "page_image_mime_type": "image/png",
                },
            },
        ]


class FakeLlmService:
    def generate(self, question: str, documents: list[RetrievedContext]) -> str:
        assert question
        assert documents
        return "Refunds are available within 30 days."

    def stream(self, question: str, documents: list[RetrievedContext]):
        yield "Refunds are available"
        yield " within 30 days."


class FakeReranker:
    def rerank(
        self,
        query: str,
        documents: list[RetrievedContext],
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        ranked = [
            RetrievedContext(
                id=documents[1].id,
                title=documents[1].title,
                content=documents[1].content,
                metadata={**documents[1].metadata, "rerank_score": 0.97},
            ),
            RetrievedContext(
                id=documents[0].id,
                title=documents[0].title,
                content=documents[0].content,
                metadata={**documents[0].metadata, "rerank_score": 0.91},
            ),
        ]
        return ranked[:top_k]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        RAG_TOP_K=2,
        RAG_RETRIEVAL_TOP_K=4,
        RETRIEVAL_MODE="hybrid",
        reranker_provider=None,
    )


def test_run_normalizes_llm_text_and_source_media() -> None:
    pipeline = RagInferencePipeline(
        vector_store=FakeVectorStore(),
        llm_service=FakeLlmService(),
        settings=_settings(),
    )

    response = pipeline.run("How do refunds work?")

    assert response["answer"] == "Refunds are available within 30 days."
    assert response["content"][0] == {
        "type": "text",
        "text": "Refunds are available within 30 days.",
    }
    assert response["sources"][0]["media"] == [
        {
            "type": "image",
            "url": "https://example.test/refund-flow.png",
            "source_id": "chunk-1",
        }
    ]


def test_run_applies_optional_reranker_scores() -> None:
    pipeline = RagInferencePipeline(
        vector_store=FakeVectorStore(),
        llm_service=FakeLlmService(),
        reranker=FakeReranker(),
        settings=_settings(),
    )

    response = pipeline.run("How do refunds work?", top_k=1)

    assert response["sources"][0]["id"] == "chunk-2"
    assert response["sources"][0]["metadata"]["rerank_score"] == 0.97
    assert response["retrieval"]["documents"] == 1


def test_visual_questions_keep_retrieved_image_context() -> None:
    vector_store = FakeVisualVectorStore()
    pipeline = RagInferencePipeline(
        vector_store=vector_store,
        llm_service=FakeLlmService(),
        settings=_settings(),
    )

    response = pipeline.run(
        "In the schedule image, what color is the far-right End column?"
    )

    assert vector_store.last_score_threshold == 0.0
    assert any(source["media"] for source in response["sources"])
    assert response["sources"][-1]["id"] == "pdf-page-image"


def test_stream_delta_empty_chunk_does_not_use_fallback_answer() -> None:
    response_builder = RagResponseBuilder()

    response = response_builder.build_delta("")

    assert response["answer"] == ""
    assert response["content"] == [{"type": "text", "text": ""}]


def test_media_extractor_ignores_invalid_media_dictionaries() -> None:
    media_extractor = MediaExtractor()

    media = media_extractor.extract(
        {
            "media": {"foo": "bar"},
            "attachments": [{"type": "image"}],
            "image": {"url": "https://example.test/valid.png"},
        },
        source_id="chunk-1",
    )

    assert media == [
        {
            "type": "image",
            "url": "https://example.test/valid.png",
            "source_id": "chunk-1",
        }
    ]
