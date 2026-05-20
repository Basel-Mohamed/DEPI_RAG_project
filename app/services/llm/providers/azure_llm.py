from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.services.llm.prompts import (
    build_context_text,
    build_langchain_chain,
    build_support_prompt,
    build_prompt_messages,
    stream_langchain_chain,
)
from app.services.llm.providers.base_llm import (
    DEFAULT_FALLBACK_ANSWER,
    BaseLlmService,
    LlmServiceError,
)
from app.services.types import RetrievedContext


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
        max_image_inputs: int = 4,
    ) -> None:
        self.azure_endpoint = azure_endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_answer = fallback_answer
        self.max_image_inputs = max_image_inputs

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
        images = self._extract_image_urls(documents)
        if not images:
            return build_prompt_messages(question=question, context=context)

        formatted_messages = build_support_prompt().format_messages(
            question=question.strip(),
            context=context or "No relevant context provided.",
        )
        return [
            {
                "role": str(formatted_messages[0].type),
                "content": str(formatted_messages[0].content),
            },
            {
                "role": str(formatted_messages[1].type),
                "content": [
                    {
                        "type": "text",
                        "text": str(formatted_messages[1].content),
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        for image_url in images
                    ],
                ],
            },
        ]

    def generate(self, question: str, documents: list[RetrievedContext]) -> str:
        if not question.strip():
            return self.fallback_answer

        context = self.build_context(documents)
        if not context:
            return self.fallback_answer

        try:
            images = self._extract_image_urls(documents)
            if images:
                answer = self._get_llm().invoke(
                    self._build_langchain_multimodal_messages(
                        question=question,
                        context=context,
                        image_urls=images,
                    )
                )
                return self._extract_text(answer) or self.fallback_answer

            answer = build_langchain_chain(self._get_llm()).invoke(
                {
                    "question": question.strip(),
                    "context": context,
                }
            )
        except Exception as exc:
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
            images = self._extract_image_urls(documents)
            if images:
                messages = self._build_langchain_multimodal_messages(
                    question=question,
                    context=context,
                    image_urls=images,
                )
                for chunk in self._get_llm().stream(messages):
                    text = self._extract_text(chunk, strip=False)
                    if text:
                        yielded = True
                        yield text
            else:
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
            raise LlmServiceError(
                "Azure OpenAI answer streaming failed."
            ) from exc

    def _build_langchain_multimodal_messages(
        self,
        *,
        question: str,
        context: str,
        image_urls: list[str],
    ) -> list[Any]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise LlmServiceError(
                "The 'langchain-core' package is required to use Azure multimodal prompts."
            ) from exc

        formatted_messages = build_support_prompt().format_messages(
            question=question.strip(),
            context=context or "No relevant context provided.",
        )
        return [
            SystemMessage(content=str(formatted_messages[0].content)),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": str(formatted_messages[1].content),
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        for image_url in image_urls
                    ],
                ]
            ),
        ]

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
                        if not self._is_image_metadata_key(key)
                    },
                )
            )
        return cleaned_documents

    def _extract_image_urls(self, documents: list[RetrievedContext]) -> list[str]:
        image_urls: list[str] = []
        seen: set[str] = set()
        for document in documents:
            for image_url in self._iter_metadata_images(document.metadata):
                if image_url in seen:
                    continue
                seen.add(image_url)
                image_urls.append(image_url)
                if len(image_urls) >= self.max_image_inputs:
                    return image_urls
        return image_urls

    def _iter_metadata_images(self, metadata: dict[str, Any]) -> Iterator[str]:
        mime_type = str(metadata.get("page_image_mime_type") or "image/png")
        for key in (
            "page_image_base64",
            "page_image_url",
            "image_base64",
            "image_url",
            "image",
            "images",
            "media",
            "attachments",
        ):
            if key not in metadata:
                continue
            yield from self._iter_image_value(metadata[key], mime_type=mime_type)

    def _iter_image_value(self, value: Any, *, mime_type: str) -> Iterator[str]:
        if value is None:
            return

        if isinstance(value, bytes):
            import base64

            encoded = base64.b64encode(value).decode("ascii")
            yield f"data:{mime_type};base64,{encoded}"
            return

        if isinstance(value, str):
            yield self._normalize_image_url(value, mime_type=mime_type)
            return

        if isinstance(value, dict):
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            candidate = (
                image_url
                or value.get("url")
                or value.get("data")
                or value.get("base64")
                or value.get("b64_json")
            )
            if candidate:
                yield self._normalize_image_url(str(candidate), mime_type=mime_type)
            return

        if isinstance(value, list):
            for item in value:
                yield from self._iter_image_value(item, mime_type=mime_type)

    @staticmethod
    def _normalize_image_url(value: str, *, mime_type: str) -> str:
        stripped = value.strip()
        if stripped.startswith(("data:", "http://", "https://")):
            return stripped
        return f"data:{mime_type};base64,{stripped}"

    @staticmethod
    def _is_image_metadata_key(key: str) -> bool:
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
                raise LlmServiceError(
                    "AzureLlmService singleton was already initialized with a different configuration."
                )
            return self.__class__._singleton_llm

        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise LlmServiceError(
                "The 'langchain-openai' package is required to use "
                "AzureLlmService."
            ) from exc

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
