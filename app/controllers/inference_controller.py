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

        logger.info("inference started mode=%s", request.mode)
        response = self.pipeline.run(
            question,
            top_k=request.top_k,
            mode=request.mode,
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

        logger.info("inference stream started mode=%s", request.mode)
        return self._stream_chunks(
            request,
            question=question,
        )

    def _stream_chunks(
        self,
        request: InferenceRequest,
        *,
        question: str,
    ) -> Iterator[str]:
        try:
            for chunk in self.pipeline.stream(
                question,
                top_k=request.top_k,
                mode=request.mode,
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
