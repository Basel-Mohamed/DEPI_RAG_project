from __future__ import annotations

import os
import unittest
from importlib import import_module

from app.services.types import RetrievedContext
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


class TestRerankerLlmIntegration(unittest.TestCase):
    def setUp(self) -> None:
        CohereRerankerService._singleton_reranker = None
        CohereRerankerService._singleton_config = None

    def _load_cohere_llm_class(self) -> type:
        try:
            cohere_module = import_module("app.services.llm.providers.cohere_llm")
        except ModuleNotFoundError as exc:
            self.skipTest(f"Missing optional LLM integration dependency: {exc}")

        cohere_cls = cohere_module.CohereLlmService
        cohere_cls._singleton_llm = None
        cohere_cls._singleton_config = None
        return cohere_cls

    def test_cohere_rerank_to_cohere_llm(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            self.skipTest("COHERE_API_KEY is not set.")

        cohere_llm_cls = self._load_cohere_llm_class()

        reranker = CohereRerankerService(
            api_key=api_key,
            model_name=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            top_n=3,
        )
        llm = cohere_llm_cls(
            api_key=api_key,
            model_name=os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025"),
            temperature=0.0,
            max_tokens=200,
        )

        query = "Explain how retrieval-augmented generation uses retrieved documents."
        chunks = _build_dummy_chunks()
        reranked = reranker.rerank(query=query, documents=chunks, top_k=3)
        answer = llm.generate(question=query, documents=reranked)

        self.assertEqual(len(reranked), 3)
        self.assertTrue(all(isinstance(chunk, RetrievedContext) for chunk in reranked))
        self.assertTrue(answer.strip())

    def test_cohere_rerank_to_cohere_llm_stream(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            self.skipTest("COHERE_API_KEY is not set.")

        cohere_llm_cls = self._load_cohere_llm_class()

        reranker = CohereRerankerService(
            api_key=api_key,
            model_name=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            top_n=3,
        )
        llm = cohere_llm_cls(
            api_key=api_key,
            model_name=os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025"),
            temperature=0.0,
            max_tokens=200,
        )

        query = "Explain how retrieval-augmented generation uses retrieved documents."
        chunks = _build_dummy_chunks()
        reranked = reranker.rerank(query=query, documents=chunks, top_k=3)
        streamed_chunks = list(llm.stream(question=query, documents=reranked))
        answer = "".join(streamed_chunks).strip()

        self.assertEqual(len(reranked), 3)
        self.assertTrue(streamed_chunks)
        self.assertTrue(answer)


if __name__ == "__main__":
    unittest.main()
