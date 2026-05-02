from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.services.llm.prompts import (
    build_context_text,
    build_langchain_chain,
    stream_langchain_chain,
)
from app.services.llm.providers.base_llm import (
    BaseLlmService,
    DEFAULT_FALLBACK_ANSWER,
    LlmServiceError,
)
from app.services.types import RetrievedContext


class CohereLlmService(BaseLlmService):
    """Cohere implementation backed by LangChain chat models."""

    _singleton_llm: Any | None = None
    _singleton_config: tuple[str, str, float, int] | None = None

    def __init__(
        self,
        api_key: str,
        model_name: str = "command-r-plus",
        *,
        temperature: float = 0.2,
        max_tokens: int = 400,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_answer = fallback_answer

    def build_context(self, documents: list[RetrievedContext]) -> str:
        """Build a concise context block for Cohere generation."""

        return build_context_text(
            documents,
            include_metadata=False,
        )

    def generate(self, question: str, documents: list[RetrievedContext]) -> str:
        """Run a standard non-streaming LangChain invocation."""

        if not question.strip():
            return self.fallback_answer

        context = self.build_context(documents)
        if not context:
            return self.fallback_answer

        try:
            answer = build_langchain_chain(self._get_llm()).invoke(
                {
                    "question": question.strip(),
                    "context": context,
                }
            )
        except Exception as exc:
            raise LlmServiceError("Cohere answer generation failed.") from exc

        return str(answer).strip() or self.fallback_answer

    def stream(self, question: str, documents: list[RetrievedContext]) -> Iterator[str]:
        """Yield streamed answer chunks from the Cohere LangChain model."""

        if not question.strip():
            yield self.fallback_answer
            return

        context = self.build_context(documents)
        if not context:
            yield self.fallback_answer
            return

        try:
            yielded = False
            for chunk in stream_langchain_chain(
                self._get_llm(),
                question=question,
                context=context,
            ):
                yielded = True
                yield chunk
            if not yielded:
                yield self.fallback_answer
        except Exception as exc:
            raise LlmServiceError("Cohere answer streaming failed.") from exc

    def _get_llm(self) -> Any:
        """Lazily create and reuse a process-wide LangChain Cohere chat model."""

        config = (
            self.api_key,
            self.model_name,
            self.temperature,
            self.max_tokens,
        )
        if self.__class__._singleton_llm is not None:
            if self.__class__._singleton_config != config:
                raise LlmServiceError(
                    "CohereLlmService singleton was already initialized with a different configuration."
                )
            return self.__class__._singleton_llm

        try:
            from langchain_cohere import ChatCohere
        except ImportError as exc:
            raise LlmServiceError(
                "The 'langchain-cohere' package is required to use "
                "CohereLlmService."
            ) from exc

        self.__class__._singleton_llm = ChatCohere(
            cohere_api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.__class__._singleton_config = config
        return self.__class__._singleton_llm
