from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from app.schemas.inference import InferenceRequest
from app.services.rag.rag_inference import RagInferencePipeline

logger = logging.getLogger(__name__)


class InferenceController:
    def __init__(self, pipeline: RagInferencePipeline) -> None:
        self.pipeline = pipeline

    def ask(self, request: InferenceRequest) -> dict[str, Any]:
        question = self._clean_question(request.question)
        filter_field, filter_value = self._filter_params(request)

        logger.info(
            "inference started mode=%s filter_field=%s",
            request.mode,
            filter_field,
        )
        response = self.pipeline.run(
            question,
            mode=request.mode,
            filter_field=filter_field,
            filter_value=filter_value,
            score_threshold=request.score_threshold,
            include_sources=request.include_sources,
        )
        logger.info(
            "inference completed documents=%s",
            response.get("retrieval", {}).get("documents"),
        )
        return response

    def stream(self, request: InferenceRequest) -> Iterator[str]:
        question = self._clean_question(request.question)
        filter_field, filter_value = self._filter_params(request)

        logger.info(
            "inference stream started mode=%s filter_field=%s",
            request.mode,
            filter_field,
        )
        return self._stream_chunks(
            request,
            question=question,
            filter_field=filter_field,
            filter_value=filter_value,
        )

    def _stream_chunks(
        self,
        request: InferenceRequest,
        *,
        question: str,
        filter_field: str | None,
        filter_value: Any,
    ) -> Iterator[str]:
        try:
            for chunk in self.pipeline.stream(
                question,
                mode=request.mode,
                filter_field=filter_field,
                filter_value=filter_value,
                score_threshold=request.score_threshold,
                include_sources=request.include_sources,
            ):
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
        finally:
            logger.info("inference stream completed")

    @staticmethod
    def _clean_question(question: str) -> str:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question is required.")
        return cleaned

    @staticmethod
    def _filter_params(request: InferenceRequest) -> tuple[str | None, Any]:
        if request.source:
            return "source", request.source
        return request.filter_field, request.filter_value
