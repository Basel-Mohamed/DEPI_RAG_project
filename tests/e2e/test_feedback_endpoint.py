import json

import pytest
from fastapi.testclient import TestClient

from app.controllers.feedback_controller import FeedbackController
from app.core.config import settings
from main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(FeedbackController, "feedback_root", tmp_path / "feedback")
    monkeypatch.setattr(
        FeedbackController,
        "feedback_path",
        tmp_path / "feedback" / "feedback.json",
    )
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_feedback_endpoint_persists_user_feedback(client):
    payload = {
        "session_id": "session-123",
        "question": "What is the refund policy?",
        "answer": "Refunds are available within 30 days.",
        "rating": 1,
        "timestamp": "2026-05-23T15:30:00Z",
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"] == "session-123"
    assert body["stored"] is True
    assert body["feedback_id"]

    stored_feedback = json.loads(FeedbackController.feedback_path.read_text(encoding="utf-8"))
    assert len(stored_feedback) == 1
    assert stored_feedback[0] == {
        "feedback_id": body["feedback_id"],
        "session_id": "session-123",
        "question": "What is the refund policy?",
        "answer": "Refunds are available within 30 days.",
        "rating": 1,
        "timestamp": "2026-05-23T15:30:00+00:00",
        "created_at": stored_feedback[0]["created_at"],
    }


def test_feedback_endpoint_rejects_invalid_rating(client):
    response = client.post(
        "/feedback",
        json={
            "session_id": "session-123",
            "question": "Question?",
            "answer": "Answer.",
            "rating": 0,
            "timestamp": "2026-05-23T15:30:00Z",
        },
    )

    assert response.status_code == 422


def test_feedback_list_endpoint_returns_saved_feedback(client):
    first_payload = {
        "session_id": "session-123",
        "question": "Question one?",
        "answer": "Answer one.",
        "rating": 1,
        "timestamp": "2026-05-23T15:30:00Z",
    }
    second_payload = {
        "session_id": "session-456",
        "question": "Question two?",
        "answer": "Answer two.",
        "rating": -1,
        "timestamp": "2026-05-23T15:31:00Z",
    }
    client.post("/feedback", json=first_payload)
    client.post("/feedback", json=second_payload)

    response = client.get("/feedback")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [item["session_id"] for item in body["feedback"]] == [
        "session-456",
        "session-123",
    ]


def test_feedback_list_endpoint_filters_by_session_and_limit(client):
    for index in range(3):
        client.post(
            "/feedback",
            json={
                "session_id": "session-123",
                "question": f"Question {index}?",
                "answer": f"Answer {index}.",
                "rating": 1,
                "timestamp": "2026-05-23T15:30:00Z",
            },
        )
    client.post(
        "/feedback",
        json={
            "session_id": "session-456",
            "question": "Other question?",
            "answer": "Other answer.",
            "rating": -1,
            "timestamp": "2026-05-23T15:31:00Z",
        },
    )

    response = client.get("/feedback?session_id=session-123&limit=2")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert all(item["session_id"] == "session-123" for item in body["feedback"])


def test_feedback_satisfaction_endpoint_returns_kpi_summary(client):
    for rating in [1, 1, -1]:
        client.post(
            "/feedback",
            json={
                "session_id": "session-123",
                "question": "Question?",
                "answer": "Answer.",
                "rating": rating,
                "timestamp": "2026-05-23T15:30:00Z",
            },
        )

    response = client.get("/feedback/satisfaction")

    assert response.status_code == 200
    assert response.json() == {
        "total": 3,
        "positive": 2,
        "negative": 1,
        "satisfaction_score": 2 / 3,
        "satisfaction_percent": 66.67,
    }


def test_feedback_satisfaction_endpoint_handles_empty_feedback(client):
    response = client.get("/feedback/satisfaction")

    assert response.status_code == 200
    assert response.json() == {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "satisfaction_score": None,
        "satisfaction_percent": None,
    }


def test_feedback_endpoint_rejects_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(FeedbackController, "feedback_root", tmp_path / "feedback")
    monkeypatch.setattr(
        FeedbackController,
        "feedback_path",
        tmp_path / "feedback" / "feedback.json",
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/feedback",
            json={
                "session_id": "session-123",
                "question": "Question?",
                "answer": "Answer.",
                "rating": -1,
                "timestamp": "2026-05-23T15:30:00Z",
            },
        )

    assert response.status_code == 401
