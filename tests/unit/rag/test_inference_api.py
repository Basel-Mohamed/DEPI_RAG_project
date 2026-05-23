from __future__ import annotations

import json
from typing import Any

import pytest

from app.controllers.inference_controller import InferenceController
from app.schemas.inference import InferenceRequest


class FakePipeline:
    def __init__(self) -> None:
        self.last_run: dict[str, Any] | None = None

    def run(self, question: str, **kwargs: Any) -> dict[str, Any]:
        self.last_run = {"question": question, **kwargs}
        return {
            "answer": "Refunds are available within 30 days.",
            "content": [
                {"type": "text", "text": "Refunds are available within 30 days."}
            ],
            "sources": [],
            "retrieval": {"documents": 1, "mode": kwargs.get("mode") or "hybrid"},
        }

    def stream(self, question: str, **kwargs: Any):
        yield {
            "event": "delta",
            "answer": "Refunds",
            "content": [{"type": "text", "text": "Refunds"}],
            "sources": [],
            "retrieval": {},
        }
        yield {
            "event": "sources",
            "answer": "",
            "content": [],
            "sources": [],
            "retrieval": {"documents": 1, "mode": kwargs.get("mode") or "hybrid"},
        }


def test_controller_ask_passes_normalized_request_to_pipeline() -> None:
    fake_pipeline = FakePipeline()
    controller = InferenceController(fake_pipeline)

    response = controller.ask(
        InferenceRequest(question=" How do refunds work? ", source="policy.pdf")
    )

    assert response["answer"] == "Refunds are available within 30 days."
    assert fake_pipeline.last_run == {
        "question": "How do refunds work?",
        "mode": None,
        "filter_field": "source",
        "filter_value": "policy.pdf",
        "score_threshold": None,
        "include_sources": True,
    }


def test_controller_stream_returns_ndjson_chunks() -> None:
    controller = InferenceController(FakePipeline())

    lines = list(controller.stream(InferenceRequest(question="How do refunds work?")))

    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "delta"
    assert json.loads(lines[1])["event"] == "sources"


def test_controller_rejects_blank_question_after_stripping() -> None:
    controller = InferenceController(FakePipeline())

    with pytest.raises(ValueError, match="Question is required"):
        controller.ask(InferenceRequest(question="   "))
