from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.rag.rag_inference import RagInferencePipeline
from app.services.types import RetrievedContext


class FakeVectorStore:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_kwargs = kwargs
        return [
            {
                "id": "chunk-1",
                "text": "Refunds are available for eligible orders within 30 days.",
                "score": 0.91,
                "metadata": {"source": "policy.pdf", "page_number": 2},
            }
        ]


class FakeLlmService:
    def generate(self, question: str, documents: list[RetrievedContext]) -> str:
        assert question == "How do refunds work?"
        assert documents[0].metadata["retrieval_score"] == 0.91
        return "Refunds are available within 30 days."


def test_rag_pipeline_retrieves_then_generates_answer() -> None:
    vector_store = FakeVectorStore()
    pipeline = RagInferencePipeline(
        vector_store=vector_store,
        llm_service=FakeLlmService(),
        settings=SimpleNamespace(
            RAG_RETRIEVAL_TOP_K=3,
            RAG_RERANK_TOP_K=1,
            RETRIEVAL_MODE="hybrid",
            reranker_provider=None,
        ),
    )

    response = pipeline.run("How do refunds work?")

    assert response["answer"] == "Refunds are available within 30 days."
    assert response["retrieval"] == {"documents": 1, "mode": "hybrid"}
    assert vector_store.last_kwargs["top_k"] == 3
    assert "filter_field" not in vector_store.last_kwargs
    assert "filter_value" not in vector_store.last_kwargs
