from __future__ import annotations

from types import SimpleNamespace

from app.services.vectorstore.qdrant_store import QdrantService, SearchMode


def test_qdrant_point_ids_are_deterministic() -> None:
    key = "policy.pdf::2::7"

    assert QdrantService._to_uuid(key) == QdrantService._to_uuid(key)
    assert QdrantService._to_uuid(key) != QdrantService._to_uuid("policy.pdf::2::8")


def test_format_scored_result_excludes_text_from_metadata() -> None:
    point = SimpleNamespace(
        id="chunk-1",
        score=0.87654,
        payload={
            "text": "Refunds are available within 30 days.",
            "source": "policy.pdf",
            "page_number": 2,
        },
    )

    result = QdrantService._format_scored(point)

    assert result == {
        "id": "chunk-1",
        "text": "Refunds are available within 30 days.",
        "score": 0.8765,
        "metadata": {"source": "policy.pdf", "page_number": 2},
    }


def test_search_mode_values_match_api_contract() -> None:
    assert {mode.value for mode in SearchMode} == {"dense", "sparse", "hybrid"}
