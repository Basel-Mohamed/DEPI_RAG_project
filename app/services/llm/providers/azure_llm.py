from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from app.services.llm.prompts import (
    build_context_text,
    build_langchain_chain,
    build_prompt_messages,
    stream_langchain_chain,
)
from app.services.llm.providers.base_llm import (
    DEFAULT_FALLBACK_ANSWER,
    BaseLlmService,
    LlmServiceError,
)
from app.services.types import RetrievedContext

logger = logging.getLogger(__name__)


class AzureLlmService(BaseLlmService):
    _singleton_llm: Any | None = None
    _singleton_config: (
        tuple[str, str, str, str, float, int] | None
    ) = None

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 400,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
    ) -> None:
        self.azure_endpoint = azure_endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_answer = fallback_answer

    def build_context(self, documents: list[RetrievedContext]) -> str:
        return build_context_text(
            self._text_only_documents(documents),
            include_metadata=True,
        )

    def build_messages(
        self,
        question: str,
        documents: list[RetrievedContext],
    ) -> list[dict[str, Any]]:
        context = self.build_context(documents)
        return build_prompt_messages(question=question, context=context)

    def generate(self, question: str, documents: list[RetrievedContext]) -> str:
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
            logger.exception("Azure OpenAI answer generation failed")
            raise LlmServiceError(
                "Azure OpenAI answer generation failed."
            ) from exc

        return str(answer).strip() or self.fallback_answer

    def stream(self, question: str, documents: list[RetrievedContext]) -> Iterator[str]:
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
            logger.exception("Azure OpenAI answer streaming failed")
            raise LlmServiceError(
                "Azure OpenAI answer streaming failed."
            ) from exc

    def _text_only_documents(
        self,
        documents: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        cleaned_documents: list[RetrievedContext] = []
        for document in documents:
            cleaned_documents.append(
                RetrievedContext(
                    id=document.id,
                    title=document.title,
                    content=document.content,
                    metadata={
                        key: value
                        for key, value in document.metadata.items()
                        if not self._is_non_text_metadata_key(key)
                    },
                )
            )
        return cleaned_documents

    @staticmethod
    def _is_non_text_metadata_key(key: str) -> bool:
        normalized = key.lower()
        return any(
            token in normalized
            for token in ("image", "images", "media", "attachment")
        )

    @staticmethod
    def _extract_text(response: Any, *, strip: bool = True) -> str:
        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content.strip() if strip else content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        text_parts.append(str(text))
            text = "".join(text_parts)
            return text.strip() if strip else text

        text = str(content)
        return text.strip() if strip else text

    def _get_llm(self) -> Any:
        config = (
            self.azure_endpoint,
            self.api_key,
            self.deployment_name,
            self.api_version,
            self.temperature,
            self.max_tokens,
        )
        if self.__class__._singleton_llm is not None:
            if self.__class__._singleton_config != config:
                logger.error("AzureLlmService singleton requested with conflicting configuration")
                raise LlmServiceError(
                    "AzureLlmService singleton was already initialized with a different configuration."
                )
            return self.__class__._singleton_llm

        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            logger.exception("langchain-openai is not installed")
            raise LlmServiceError(
                "The 'langchain-openai' package is required to use "
                "AzureLlmService."
            ) from exc

        logger.info("creating AzureChatOpenAI client deployment=%s", self.deployment_name)
        self.__class__._singleton_llm = AzureChatOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            azure_deployment=self.deployment_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.__class__._singleton_config = config
        return self.__class__._singleton_llm
