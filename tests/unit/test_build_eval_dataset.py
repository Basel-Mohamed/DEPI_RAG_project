from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from scripts import build_eval_dataset


class FakeChatModel:
    def __init__(self) -> None:
        self.passages: list[str] = []

    def invoke(self, messages):
        passage = messages[1][1].replace("Passage:\n", "")
        self.passages.append(passage)
        return json.dumps(
            {
                "question": "What does the passage say?",
                "answer": "It gives the grounded answer.",
                "context": passage,
            }
        )


class FakeLlmService:
    def __init__(self) -> None:
        self.model = FakeChatModel()

    def _get_llm(self) -> FakeChatModel:
        return self.model


def test_parse_json_response_accepts_fenced_json() -> None:
    parsed = build_eval_dataset.parse_json_response(
        '```json\n{"question":"Q","answer":"A","context":"C"}\n```'
    )

    assert parsed == {"question": "Q", "answer": "A", "context": "C"}


def test_build_dataset_samples_and_skips_short_chunks(monkeypatch) -> None:
    test_dir = Path("test_build_eval_dataset_tmp") / uuid.uuid4().hex
    corpus_path = test_dir / "corpus.jsonl"
    output_path = test_dir / "nested" / "eval.jsonl"
    long_text = "Refund requests are supported within 30 days. " * 5
    short_text = "Too short."
    chunks = [
        {"text": long_text, "metadata": {"source": "policy.md"}},
        {"text": short_text, "metadata": {"source": "tiny.md"}},
    ]
    test_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        "\n".join(json.dumps(chunk) for chunk in chunks),
        encoding="utf-8",
    )
    fake_service = FakeLlmService()
    monkeypatch.setattr(
        build_eval_dataset,
        "create_llm_service",
        lambda settings: fake_service,
    )

    try:
        summary = build_eval_dataset.build_dataset(
            corpus_path=corpus_path,
            output_path=output_path,
            samples=2,
            seed=0,
        )

        records = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        assert summary == {
            "corpus_chunks": 2,
            "sampled": 2,
            "generated": 1,
            "skipped_short": 1,
            "failed": 0,
        }
        assert records == [
            {
                "question": "What does the passage say?",
                "answer": "It gives the grounded answer.",
                "context": long_text.strip(),
            }
        ]
    finally:
        shutil.rmtree(test_dir.parents[0], ignore_errors=True)
