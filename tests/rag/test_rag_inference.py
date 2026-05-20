from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.rag.rag_inference import RagInferencePipeline
from app.services.types import RankedDocument, RetrievedContext


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


class FakeLlmService:
    def generate(self, question: str, documents: list[RetrievedContext]) -> dict[str, Any]:
        assert question == "How do refunds work?"
        assert documents
        return {
            "content": [
                {"type": "text", "text": "Refunds are available within 30 days."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ]
        }

    def stream(self, question: str, documents: list[RetrievedContext]):
        yield "Refunds are available"
        yield " within 30 days."


class FakeReranker:
    def rank_with_scores(
        self,
        query: str,
        documents: list[RetrievedContext],
        top_k: int | None = None,
    ) -> list[RankedDocument]:
        ranked = [
            RankedDocument(document=documents[1], score=0.97),
            RankedDocument(document=documents[0], score=0.91),
        ]
        return ranked[:top_k]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        RAG_TOP_K=2,
        RAG_RETRIEVAL_TOP_K=4,
        RETRIEVAL_MODE="hybrid",
        reranker_provider=None,
    )


def test_run_preserves_multimodal_llm_output_and_source_media() -> None:
    pipeline = RagInferencePipeline(
        vector_store=FakeVectorStore(),
        llm_service=FakeLlmService(),
        settings=_settings(),
    )

    response = pipeline.run("How do refunds work?")

    assert response["answer"] == "Refunds are available within 30 days."
    assert response["images"] == [
        {"type": "image", "url": "data:image/png;base64,abc123"}
    ]
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
