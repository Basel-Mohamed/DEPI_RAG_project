from __future__ import annotations

import base64
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, settings as global_settings
from app.services.llm.providers.base_llm import (
    DEFAULT_FALLBACK_ANSWER,
    BaseLlmService,
)
from app.services.types import RankedDocument, RetrievedContext

if TYPE_CHECKING:
    from app.services.reranking.base_reranker import BaseRerankerService
    from app.services.vectorstore.qdrant_store import QdrantService, SearchMode

logger = logging.getLogger(__name__)

GeneratedPayload = dict[str, Any]
SourcePayload = dict[str, Any]


class RagInferenceError(RuntimeError):
    """Raised when the RAG inference pipeline cannot complete a request."""


class RagInferencePipeline:
    """Retrieve, optionally rerank, and generate grounded answers.

    The pipeline keeps multimodal model output as structured content. Text-only
    providers produce a plain answer string, while providers that return content
    blocks, image URLs, or base64 image data expose those artifacts through the
    response's ``content`` and ``images`` fields.
    """

    def __init__(
        self,
        *,
        vector_store: QdrantService | None = None,
        llm_service: BaseLlmService | None = None,
        reranker: BaseRerankerService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or global_settings
        if vector_store is None:
            from app.services.embedding.embedding_service import EmbeddingService
            from app.services.vectorstore.qdrant_store import QdrantService

            vector_store = QdrantService(
                embedding_service=EmbeddingService(settings=self.settings),
                settings=self.settings,
            )

        if llm_service is None:
            from app.services.llm.llm_factory import create_llm_service

            llm_service = create_llm_service(self.settings)

        self.vector_store = vector_store
        self.llm_service = llm_service
        self.reranker = reranker if reranker is not None else self._create_reranker()

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retrieval_top_k: int | None = None,
        mode: SearchMode | str | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
        include_sources: bool = True,
    ) -> GeneratedPayload:
        """Return a complete RAG response for ``question``."""

        clean_question = question.strip()
        if not clean_question:
            return self._empty_response(DEFAULT_FALLBACK_ANSWER)

        documents = self.retrieve(
            clean_question,
            top_k=top_k,
            retrieval_top_k=retrieval_top_k,
            mode=mode,
            filter_field=filter_field,
            filter_value=filter_value,
            score_threshold=score_threshold,
        )

        if not documents:
            return self._empty_response(DEFAULT_FALLBACK_ANSWER)

        try:
            raw_answer = self.llm_service.generate(clean_question, documents)
        except Exception as exc:
            raise RagInferenceError("Answer generation failed.") from exc

        generated = self._normalize_generated_payload(raw_answer)
        generated["sources"] = (
            self._build_sources(documents) if include_sources else []
        )
        generated["retrieval"] = {
            "documents": len(documents),
            "mode": str(mode or getattr(self.settings, "RETRIEVAL_MODE", "hybrid")),
        }
        return generated

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retrieval_top_k: int | None = None,
        mode: SearchMode | str | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve and rerank context documents for the question."""

        clean_question = question.strip()
        if not clean_question:
            return []

        final_top_k = top_k or int(getattr(self.settings, "RAG_TOP_K", 5))
        search_top_k = retrieval_top_k or int(
            getattr(self.settings, "RAG_RETRIEVAL_TOP_K", max(final_top_k, 10))
        )
        search_top_k = max(search_top_k, final_top_k)

        try:
            raw_results = self.vector_store.search(
                query=clean_question,
                top_k=search_top_k,
                mode=self._coerce_search_mode(mode),
                filter_field=filter_field,
                filter_value=filter_value,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise RagInferenceError("Retrieval failed.") from exc

        documents = [self._result_to_context(result) for result in raw_results]
        return self._rerank(clean_question, documents, top_k=final_top_k)

    def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retrieval_top_k: int | None = None,
        mode: SearchMode | str | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
        include_sources: bool = True,
    ) -> Iterator[GeneratedPayload]:
        """Yield normalized answer deltas followed by source metadata."""

        clean_question = question.strip()
        if not clean_question:
            yield self._empty_response(DEFAULT_FALLBACK_ANSWER)
            return

        documents = self.retrieve(
            clean_question,
            top_k=top_k,
            retrieval_top_k=retrieval_top_k,
            mode=mode,
            filter_field=filter_field,
            filter_value=filter_value,
            score_threshold=score_threshold,
        )
        if not documents:
            yield self._empty_response(DEFAULT_FALLBACK_ANSWER)
            return

        try:
            for raw_chunk in self.llm_service.stream(clean_question, documents):
                chunk = self._normalize_generated_payload(raw_chunk)
                chunk["event"] = "delta"
                yield chunk
        except Exception as exc:
            raise RagInferenceError("Answer streaming failed.") from exc

        yield {
            "event": "sources",
            "answer": "",
            "content": [],
            "images": [],
            "sources": self._build_sources(documents) if include_sources else [],
            "retrieval": {
                "documents": len(documents),
                "mode": str(mode or getattr(self.settings, "RETRIEVAL_MODE", "hybrid")),
            },
        }

    def _create_reranker(self) -> BaseRerankerService | None:
        provider = (getattr(self.settings, "reranker_provider", None) or "").lower()
        if provider in {"", "none", "off", "false", "disabled"}:
            return None

        from app.services.reranking.reranker_factory import RerankerFactory

        top_n = getattr(self.settings, "reranker_top_n", None)
        if provider == "cohere":
            api_key = getattr(self.settings, "cohere_api_key", None)
            if not api_key:
                logger.warning("Cohere reranker requested but COHERE_API_KEY is missing.")
                return None
            return RerankerFactory.create_cohere_reranker(
                api_key=api_key,
                model_name=getattr(self.settings, "cohere_rerank_model", "rerank-v3.5"),
                top_n=top_n,
            )

        if provider == "azure_cohere":
            api_key = getattr(self.settings, "azure_cohere_api_key", None)
            base_url = getattr(self.settings, "azure_cohere_base_url", None)
            if not api_key or not base_url:
                logger.warning(
                    "Azure Cohere reranker requested but API key or base URL is missing."
                )
                return None
            return RerankerFactory.create_azure_cohere_reranker(
                api_key=api_key,
                base_url=base_url,
                model_name=getattr(self.settings, "azure_cohere_model", "model"),
                top_n=top_n,
            )

        logger.warning("Unsupported reranker provider '%s'; reranking disabled.", provider)
        return None

    def _rerank(
        self,
        question: str,
        documents: list[RetrievedContext],
        *,
        top_k: int,
    ) -> list[RetrievedContext]:
        if not documents or top_k <= 0:
            return []
        if self.reranker is None:
            return documents[:top_k]

        try:
            ranked = self.reranker.rank_with_scores(question, documents, top_k=top_k)
        except Exception:
            logger.exception("Reranking failed; falling back to retrieval order.")
            return documents[:top_k]

        return [self._attach_rerank_score(item) for item in ranked]

    @staticmethod
    def _attach_rerank_score(item: RankedDocument) -> RetrievedContext:
        metadata = {**item.document.metadata, "rerank_score": item.score}
        return RetrievedContext(
            id=item.document.id,
            title=item.document.title,
            content=item.document.content,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_search_mode(mode: SearchMode | str | None) -> SearchMode | str | None:
        if mode is None:
            return mode
        mode_value = getattr(mode, "value", mode)
        normalized = str(mode_value).lower()
        if normalized not in {"dense", "sparse", "hybrid"}:
            raise ValueError("Search mode must be one of: dense, sparse, hybrid.")
        return normalized

    @staticmethod
    def _result_to_context(result: Mapping[str, Any]) -> RetrievedContext:
        metadata = dict(result.get("metadata") or {})
        score = result.get("score")
        if score is not None:
            metadata["retrieval_score"] = score

        source = str(metadata.get("source") or "unknown source")
        page_number = metadata.get("page_number")
        title = str(metadata.get("title") or source)
        if page_number not in (None, ""):
            title = f"{title} p.{page_number}"

        return RetrievedContext(
            id=str(result.get("id") or metadata.get("id") or ""),
            title=title,
            content=str(result.get("text") or result.get("content") or ""),
            metadata=metadata,
        )

    def _build_sources(self, documents: list[RetrievedContext]) -> list[SourcePayload]:
        sources: list[SourcePayload] = []
        for index, document in enumerate(documents, start=1):
            sources.append(
                {
                    "rank": index,
                    "id": document.id,
                    "title": document.title,
                    "content": document.content,
                    "metadata": self._json_safe_metadata(document.metadata),
                    "media": self._extract_media(document.metadata, source_id=document.id),
                }
            )
        return sources

    def _normalize_generated_payload(self, payload: Any) -> GeneratedPayload:
        content: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        text_parts: list[str] = []

        self._collect_content(payload, content=content, images=images, text_parts=text_parts)
        answer = "\n".join(part for part in text_parts if part).strip()
        if not content and answer:
            content.append({"type": "text", "text": answer})

        return {
            "answer": answer or DEFAULT_FALLBACK_ANSWER,
            "content": content,
            "images": images,
            "sources": [],
            "retrieval": {},
        }

    def _collect_content(
        self,
        payload: Any,
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
        text_parts: list[str],
    ) -> None:
        if payload is None:
            return

        if isinstance(payload, str):
            self._add_text(payload, content=content, text_parts=text_parts)
            return

        if isinstance(payload, bytes):
            self._add_image(
                {"data": base64.b64encode(payload).decode("ascii")},
                content=content,
                images=images,
            )
            return

        if isinstance(payload, Mapping):
            if self._try_collect_mapping(
                payload,
                content=content,
                images=images,
                text_parts=text_parts,
            ):
                return
            self._add_text(str(payload), content=content, text_parts=text_parts)
            return

        if isinstance(payload, Sequence):
            for item in payload:
                self._collect_content(
                    item,
                    content=content,
                    images=images,
                    text_parts=text_parts,
                )
            return

        if hasattr(payload, "content"):
            self._collect_content(
                getattr(payload, "content"),
                content=content,
                images=images,
                text_parts=text_parts,
            )
            return

        self._add_text(str(payload), content=content, text_parts=text_parts)

    def _try_collect_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
        text_parts: list[str],
    ) -> bool:
        handled = False
        item_type = str(payload.get("type") or "").lower()

        if item_type in {"text", "output_text", "input_text"}:
            text = payload.get("text") or payload.get("content") or payload.get("value")
            if text is not None:
                self._add_text(str(text), content=content, text_parts=text_parts)
                handled = True

        if item_type in {"image", "image_url", "input_image", "output_image"}:
            self._add_image(payload, content=content, images=images)
            handled = True

        if handled and item_type:
            return True

        for key in ("answer", "text", "output_text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                self._add_text(value, content=content, text_parts=text_parts)
                handled = True
                break

        for key in ("content", "parts", "output"):
            if key in payload and not isinstance(payload.get(key), str):
                self._collect_content(
                    payload[key],
                    content=content,
                    images=images,
                    text_parts=text_parts,
                )
                handled = True

        for key in ("image", "images", "image_url", "image_urls", "media", "attachments"):
            if key in payload:
                self._collect_media_value(
                    payload[key],
                    content=content,
                    images=images,
                )
                handled = True

        if "b64_json" in payload or "base64" in payload or "data" in payload:
            self._add_image(payload, content=content, images=images)
            handled = True

        return handled

    @staticmethod
    def _add_text(
        text: str,
        *,
        content: list[dict[str, Any]],
        text_parts: list[str],
    ) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        text_parts.append(clean_text)
        content.append({"type": "text", "text": clean_text})

    def _collect_media_value(
        self,
        value: Any,
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> None:
        if value is None:
            return
        if isinstance(value, bytes):
            self._add_image(
                {"data": base64.b64encode(value).decode("ascii")},
                content=content,
                images=images,
            )
            return
        if isinstance(value, str):
            self._add_image({"url": value}, content=content, images=images)
            return
        if isinstance(value, Mapping):
            self._add_image(value, content=content, images=images)
            return
        if isinstance(value, Sequence):
            for item in value:
                self._collect_media_value(item, content=content, images=images)

    def _add_image(
        self,
        payload: Mapping[str, Any],
        *,
        content: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> None:
        image_url = payload.get("image_url")
        if isinstance(image_url, Mapping):
            image_url = image_url.get("url")

        image = {
            "type": "image",
            "url": image_url or payload.get("url"),
            "data": (
                payload.get("data")
                or payload.get("base64")
                or payload.get("b64_json")
                or payload.get("result")
            ),
            "mime_type": payload.get("mime_type") or payload.get("media_type"),
            "metadata": dict(payload.get("metadata") or {}),
        }
        image = {key: value for key, value in image.items() if value not in (None, {}, "")}
        if not image.get("url") and not image.get("data"):
            return
        images.append(image)
        content.append(image)

    def _extract_media(
        self,
        metadata: Mapping[str, Any],
        *,
        source_id: str,
    ) -> list[dict[str, Any]]:
        media: list[dict[str, Any]] = []
        for key in (
            "image",
            "images",
            "image_url",
            "image_urls",
            "image_base64",
            "page_image",
            "page_image_base64",
            "page_image_url",
            "media",
            "attachments",
        ):
            if key not in metadata:
                continue
            self._collect_source_media(metadata[key], media=media, source_id=source_id)
        return media

    def _collect_source_media(
        self,
        value: Any,
        *,
        media: list[dict[str, Any]],
        source_id: str,
    ) -> None:
        if value is None:
            return
        if isinstance(value, bytes):
            media.append(
                {
                    "type": "image",
                    "data": base64.b64encode(value).decode("ascii"),
                    "source_id": source_id,
                }
            )
            return
        if isinstance(value, str):
            media.append({"type": "image", "url": value, "source_id": source_id})
            return
        if isinstance(value, Mapping):
            image: dict[str, Any] = {
                "type": value.get("type") or "image",
                "url": value.get("url") or value.get("image_url"),
                "data": value.get("data") or value.get("base64") or value.get("b64_json"),
                "mime_type": value.get("mime_type") or value.get("media_type"),
                "source_id": source_id,
            }
            media.append(
                {key: item for key, item in image.items() if item not in (None, "")}
            )
            return
        if isinstance(value, Sequence):
            for item in value:
                self._collect_source_media(item, media=media, source_id=source_id)

    @staticmethod
    def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, bytes):
                safe[key] = base64.b64encode(value).decode("ascii")
            else:
                safe[key] = value
        return safe

    @staticmethod
    def _empty_response(answer: str) -> GeneratedPayload:
        return {
            "answer": answer,
            "content": [{"type": "text", "text": answer}],
            "images": [],
            "sources": [],
            "retrieval": {"documents": 0},
        }


RagInferenceService = RagInferencePipeline
