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
                "metadata": {"source": "policy.pdf", "page_number": 2},
            },
            {
                "id": "chunk-2",
                "text": "Customers need the order id to start the refund workflow.",
                "score": 0.68,
                "metadata": {"source": "policy.pdf", "page_number": 3},
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
        RAG_RETRIEVAL_TOP_K=4,
        RAG_RERANK_TOP_K=2,
        RETRIEVAL_MODE="hybrid",
        reranker_provider=None,
    )


def test_run_normalizes_llm_text_and_sources() -> None:
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


def test_run_applies_optional_reranker_scores() -> None:
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


def test_stream_delta_empty_chunk_does_not_use_fallback_answer() -> None:
    response_builder = RagResponseBuilder()

    response = response_builder.build_delta("")

    assert response["answer"] == ""
    assert response["content"] == [{"type": "text", "text": ""}]
