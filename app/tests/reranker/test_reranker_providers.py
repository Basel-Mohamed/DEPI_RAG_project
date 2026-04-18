from __future__ import annotations

import os
import unittest

from app.services.types import RetrievedContext
from app.services.reranker.providers.azure_reranker import AzureCohereRerankerService
from app.services.reranker.providers.cohere_reranker import CohereRerankerService


def _build_dummy_chunks() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            id="chunk-1",
            title="Weather Notes",
            content="This chunk talks about sunshine and temperature forecasts.",
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-2",
            title="RAG Overview",
            content=(
                "Retrieval-augmented generation uses retrieved documents as context "
                "before an LLM answers a query."
            ),
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-3",
            title="Chunking",
            content="Chunking splits long documents into smaller searchable units.",
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-4",
            title="Embeddings",
            content="Embeddings map text into vectors for semantic similarity tasks.",
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-5",
            title="Reranking",
            content="Reranking reorders retrieved passages so the most relevant results appear first.",
            metadata={"source": "dummy"},
        ),
    ]


class TestRerankerProviders(unittest.TestCase):
    def setUp(self) -> None:
        CohereRerankerService._singleton_reranker = None
        CohereRerankerService._singleton_config = None
        AzureCohereRerankerService._singleton_reranker = None
        AzureCohereRerankerService._singleton_config = None

    def test_cohere(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            self.skipTest("COHERE_API_KEY is not set.")

        service = CohereRerankerService(
            api_key=api_key,
            model_name=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            top_n=5,
        )

        query = "How does retrieval-augmented generation use retrieved documents?"
        chunks = _build_dummy_chunks()
        reranked = service.rerank(query=query, documents=chunks, top_k=5)

        self.assertEqual(len(reranked), 5)
        self.assertTrue(all(isinstance(chunk, RetrievedContext) for chunk in reranked))
        self.assertIn(reranked[0].id, {"chunk-2", "chunk-5"})

    def test_azure(self) -> None:
        api_key = os.getenv("AZURE_COHERE_API_KEY")
        base_url = os.getenv("AZURE_COHERE_BASE_URL")

        if not api_key or not base_url:
            self.skipTest(
                "AZURE_COHERE_API_KEY or AZURE_COHERE_BASE_URL is not set."
            )

        service = AzureCohereRerankerService(
            api_key=api_key,
            base_url=base_url,
            model_name=os.getenv("AZURE_COHERE_MODEL", "model"),
            top_n=5,
        )

        query = "Which chunk explains reranking or retrieved documents in RAG?"
        chunks = _build_dummy_chunks()
        reranked = service.rerank(query=query, documents=chunks, top_k=5)

        self.assertEqual(len(reranked), 5)
        self.assertTrue(all(isinstance(chunk, RetrievedContext) for chunk in reranked))
        self.assertIn(reranked[0].id, {"chunk-2", "chunk-5"})


if __name__ == "__main__":
    unittest.main()
