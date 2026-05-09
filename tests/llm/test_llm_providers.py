from __future__ import annotations

import os
import unittest
from importlib import import_module

from app.services.types import RetrievedContext


def _build_dummy_chunks() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            id="chunk-1",
            title="Machine Learning Basics",
            content="Machine learning lets systems learn patterns from data.",
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-2",
            title="RAG Architecture",
            content=(
                "Retrieval-augmented generation combines retrieval with an LLM so the "
                "answer is grounded in external documents."
            ),
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-3",
            title="Vector Databases",
            content="Vector stores index embeddings and support similarity search.",
            metadata={"source": "dummy"},
        ),
        RetrievedContext(
            id="chunk-4",
            title="Chunking Strategy",
            content="Chunking splits long documents into smaller passages for retrieval.",
            metadata={"source": "dummy"},
        ),
    ]


class TestLlmProviders(unittest.TestCase):
    def _load_provider_classes(self) -> tuple[type, type]:
        try:
            azure_module = import_module("app.services.llm.providers.azure_llm")
            cohere_module = import_module("app.services.llm.providers.cohere_llm")
        except ModuleNotFoundError as exc:
            self.skipTest(f"Missing optional LLM test dependency: {exc}")

        azure_cls = azure_module.AzureLlmService
        cohere_cls = cohere_module.CohereLlmService
        azure_cls._singleton_llm = None
        azure_cls._singleton_config = None
        cohere_cls._singleton_llm = None
        cohere_cls._singleton_config = None
        return azure_cls, cohere_cls

    def test_cohere(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            self.skipTest("COHERE_API_KEY is not set.")

        _, cohere_cls = self._load_provider_classes()
                                                                                                        
        service = cohere_cls(
            api_key=api_key,
            model_name=os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025"),
            temperature=0.0,
            max_tokens=200,
        )

        query = "What is retrieval-augmented generation and why is it useful?"
        chunks = _build_dummy_chunks()
        answer = service.generate(question=query, documents=chunks)
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())

    def test_cohere_stream(self) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            self.skipTest("COHERE_API_KEY is not set.")

        _, cohere_cls = self._load_provider_classes()

        service = cohere_cls(
            api_key=api_key,
            model_name=os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025"),
            temperature=0.0,
            max_tokens=200,
        )

        query = "What is retrieval-augmented generation and why is it useful?"
        chunks = _build_dummy_chunks()
        streamed_chunks = list(service.stream(question=query, documents=chunks))
        answer = "".join(streamed_chunks).strip()

        self.assertTrue(streamed_chunks)
        self.assertTrue(answer)

    def test_azure(self) -> None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not endpoint or not api_key or not deployment:
            self.skipTest(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, or "
                "AZURE_OPENAI_CHAT_DEPLOYMENT is not set."
            )

        azure_cls, _ = self._load_provider_classes()

        service = azure_cls(
            azure_endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment,
            api_version=api_version,
            temperature=0.0,
            max_tokens=200,
        )

        query = "Explain retrieval-augmented generation in one short paragraph."
        chunks = _build_dummy_chunks()
        answer = service.generate(question=query, documents=chunks)
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())

    def test_azure_stream(self) -> None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not endpoint or not api_key or not deployment:
            self.skipTest(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, or "
                "AZURE_OPENAI_CHAT_DEPLOYMENT is not set."
            )

        azure_cls, _ = self._load_provider_classes()

        service = azure_cls(
            azure_endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment,
            api_version=api_version,
            temperature=0.0,
            max_tokens=200,
        )

        query = "Explain retrieval-augmented generation in one short paragraph."
        chunks = _build_dummy_chunks()
        streamed_chunks = list(service.stream(question=query, documents=chunks))
        answer = "".join(streamed_chunks).strip()

        self.assertTrue(streamed_chunks)
        self.assertTrue(answer)


if __name__ == "__main__":
    unittest.main()
