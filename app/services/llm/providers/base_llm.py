from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from app.services.types import RetrievedContext


DEFAULT_FALLBACK_ANSWER = (
    "I couldn't find enough verified information in the provided support knowledge "
    "base to answer that confidently."
)


class LlmServiceError(RuntimeError):
    """Raised when an LLM provider cannot generate or stream a response."""

    pass


class BaseLlmService(ABC):
    """Common interface for all LLM providers used in the answer pipeline."""

    @abstractmethod
    def build_context(self, documents: list[RetrievedContext]) -> str:
        """Convert retrieved documents into a provider-ready context string."""

        raise NotImplementedError

    @abstractmethod
    def generate(self, question: str, documents: list[RetrievedContext]) -> str:
        """Return a complete answer for the question using the given documents."""

        raise NotImplementedError

    @abstractmethod
    def stream(self, question: str, documents: list[RetrievedContext]) -> Iterator[str]:
        """Yield answer chunks incrementally for streaming clients."""

        raise NotImplementedError

    def generate_answer(self, question: str, documents: list[RetrievedContext]) -> str:
        """Backward-compatible alias used by the current RAG service."""

        return self.generate(question=question, documents=documents)
