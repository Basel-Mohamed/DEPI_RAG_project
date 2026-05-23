from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from app.schemas.inference import InferenceRequest
from app.services.llm.providers.base_llm import DEFAULT_FALLBACK_ANSWER
from app.services.monitoring.grafana_service import metrics_service
from app.services.rag.rag_inference import RagInferencePipeline

logger = logging.getLogger(__name__)
STREAM_ERROR_MESSAGE = "Inference failed. Check server logs for the request id."


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
        start = time.perf_counter()
        metrics_service.increment("inference_requests_total")
        try:
            response = self.pipeline.run(
                question,
                mode=request.mode,
                filter_field=filter_field,
                filter_value=filter_value,
                score_threshold=request.score_threshold,
                include_sources=request.include_sources,
            )
        except Exception:
            metrics_service.increment("inference_errors_total")
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        metrics_service.observe_latency("inference_ms", duration_ms)
        documents = int(response.get("retrieval", {}).get("documents") or 0)
        metrics_service.increment("retrieved_documents_total", documents)
        if response.get("answer") == DEFAULT_FALLBACK_ANSWER:
            metrics_service.increment("fallback_answers_total")
        logger.info(
            "inference completed documents=%s",
            documents,
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
        metrics_service.increment("inference_requests_total")
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
            start = time.perf_counter()
            for chunk in self.pipeline.stream(
                question,
                mode=request.mode,
                filter_field=filter_field,
                filter_value=filter_value,
                score_threshold=request.score_threshold,
                include_sources=request.include_sources,
            ):
                documents = int(chunk.get("retrieval", {}).get("documents") or 0)
                if documents:
                    metrics_service.increment("retrieved_documents_total", documents)
                if chunk.get("answer") == DEFAULT_FALLBACK_ANSWER:
                    metrics_service.increment("fallback_answers_total")
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
            metrics_service.observe_latency(
                "inference_ms",
                round((time.perf_counter() - start) * 1000, 2),
            )
        except Exception:
            metrics_service.increment("inference_errors_total")
            logger.exception("inference stream failed")
            yield json.dumps(
                {
                    "event": "error",
                    "answer": "",
                    "content": [{"type": "text", "text": STREAM_ERROR_MESSAGE}],
                    "sources": [],
                    "retrieval": {},
                },
                ensure_ascii=False,
            ) + "\n"
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
