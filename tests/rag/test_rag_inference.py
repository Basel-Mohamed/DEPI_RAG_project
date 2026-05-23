from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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


class RecordingVectorStore(FakeVectorStore):
    def __init__(self) -> None:
        self.last_top_k: int | None = None
        self.last_score_threshold: float | None = None

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_top_k = kwargs.get("top_k")
        self.last_score_threshold = kwargs.get("score_threshold")
        return super().search(**kwargs)


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
        RERANKED_CONTEXT_TOP_K=2,
        RETRIEVAL_CANDIDATE_TOP_K=4,
        RETRIEVAL_MODE="hybrid",
        reranker_provider=None,
    )


def test_run_normalizes_llm_text_and_text_sources() -> None:
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
    assert "media" not in response["sources"][0]
    assert "image_url" not in response["sources"][0]["metadata"]


def test_run_uses_settings_for_rerank_top_k() -> None:
    pipeline = RagInferencePipeline(
        vector_store=FakeVectorStore(),
        llm_service=FakeLlmService(),
        reranker=FakeReranker(),
        settings=_settings(),
    )

    response = pipeline.run("How do refunds work?")

    assert response["sources"][0]["id"] == "chunk-2"
    assert response["sources"][0]["metadata"]["rerank_score"] == 0.97
    assert response["retrieval"]["documents"] == 2


def test_retrieve_uses_settings_for_retrieval_top_k() -> None:
    vector_store = RecordingVectorStore()
    settings = SimpleNamespace(
        RERANKED_CONTEXT_TOP_K=2,
        RETRIEVAL_CANDIDATE_TOP_K=7,
        RETRIEVAL_MODE="hybrid",
        reranker_provider=None,
    )
    pipeline = RagInferencePipeline(
        vector_store=vector_store,
        llm_service=FakeLlmService(),
        settings=settings,
    )

    pipeline.retrieve("How do refunds work?")

    assert vector_store.last_top_k == 7


def test_visual_words_do_not_change_text_retrieval_threshold() -> None:
    vector_store = RecordingVectorStore()
    pipeline = RagInferencePipeline(
        vector_store=vector_store,
        llm_service=FakeLlmService(),
        settings=_settings(),
    )

    pipeline.run("In the schedule image, what text appears in the End column?")

    assert vector_store.last_score_threshold is None


def test_stream_delta_empty_chunk_does_not_use_fallback_answer() -> None:
    response_builder = RagResponseBuilder()

    response = response_builder.build_delta("")

    assert response["answer"] == ""
    assert response["content"] == [{"type": "text", "text": ""}]


def test_response_metadata_redacts_non_text_fields() -> None:
    builder = RagResponseBuilder()

    sources = builder.build_sources(
        [
            RetrievedContext(
                id="chunk-1",
                title="Policy p.1",
                content="Text",
                metadata={
                    "source": "policy.pdf",
                    "page_image_path": "uploads/page_images/file-123/1.png",
                    "page_image_url": "/media/page-images/file-123/1.png",
                    "page_image_mime_type": "image/png",
                    "attachment": "receipt.png",
                },
            )
        ]
    )

    assert sources[0] == {
        "rank": 1,
        "id": "chunk-1",
        "title": "Policy p.1",
        "content": "Text",
        "metadata": {"source": "policy.pdf"},
    }
