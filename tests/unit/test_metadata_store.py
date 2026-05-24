from app.services.metadata_store import SqliteMetadataStore


def test_sqlite_metadata_store_persists_file_registry(tmp_path):
    store = SqliteMetadataStore(tmp_path / "metadata.sqlite3")
    registry_path = tmp_path / "files.json"
    registry = {
        "file-1": {
            "file_id": "file-1",
            "filename": "sample.pdf",
            "status": "uploaded",
            "created_at": "2026-05-23T10:00:00+00:00",
            "updated_at": "2026-05-23T10:00:00+00:00",
        }
    }

    store.write_file_registry(registry, registry_path)

    assert store.read_file_registry(registry_path) == registry


def test_sqlite_metadata_store_appends_feedback(tmp_path):
    store = SqliteMetadataStore(tmp_path / "metadata.sqlite3")
    feedback_path = tmp_path / "feedback.json"
    first = {
        "feedback_id": "feedback-1",
        "session_id": "session-1",
        "question": "Question?",
        "answer": "Answer.",
        "rating": 1,
        "timestamp": "2026-05-23T10:00:00+00:00",
        "created_at": "2026-05-23T10:00:01+00:00",
    }
    second = {
        "feedback_id": "feedback-2",
        "session_id": "session-2",
        "question": "Another question?",
        "answer": "Another answer.",
        "rating": -1,
        "timestamp": "2026-05-23T11:00:00+00:00",
        "created_at": "2026-05-23T11:00:01+00:00",
    }

    store.append_feedback(first, feedback_path)
    store.append_feedback(second, feedback_path)

    assert store.read_feedback(feedback_path) == [second, first]


def test_sqlite_metadata_store_clears_feedback(tmp_path):
    store = SqliteMetadataStore(tmp_path / "metadata.sqlite3")
    feedback_path = tmp_path / "feedback.json"
    store.append_feedback(
        {
            "feedback_id": "feedback-1",
            "session_id": "session-1",
            "question": "Question?",
            "answer": "Answer.",
            "rating": 1,
            "timestamp": "2026-05-23T10:00:00+00:00",
            "created_at": "2026-05-23T10:00:01+00:00",
        },
        feedback_path,
    )
    store.append_feedback(
        {
            "feedback_id": "feedback-2",
            "session_id": "session-2",
            "question": "Another question?",
            "answer": "Another answer.",
            "rating": -1,
            "timestamp": "2026-05-23T11:00:00+00:00",
            "created_at": "2026-05-23T11:00:01+00:00",
        },
        feedback_path,
    )

    assert store.clear_feedback(feedback_path) == 2
    assert store.read_feedback(feedback_path) == []


def test_sqlite_metadata_store_persists_and_clears_monitoring_metrics(tmp_path):
    store = SqliteMetadataStore(tmp_path / "metadata.sqlite3")
    metrics_path = tmp_path / "metrics.json"
    first = {
        "metric_id": "metric-1",
        "metric_name": "request_latency_ms",
        "value": 50.0,
        "created_at": "2026-05-23T10:00:00+00:00",
    }
    second = {
        "metric_id": "metric-2",
        "metric_name": "llm_tokens",
        "value": 25,
        "created_at": "2026-05-23T10:00:01+00:00",
    }

    store.append_monitoring_metric(first, metrics_path)
    store.append_monitoring_metric(second, metrics_path)

    assert store.read_monitoring_metrics(metrics_path) == [
        {
            "metric_name": "request_latency_ms",
            "value": 50.0,
            "created_at": "2026-05-23T10:00:00+00:00",
        },
        {
            "metric_name": "llm_tokens",
            "value": 25.0,
            "created_at": "2026-05-23T10:00:01+00:00",
        },
    ]
    assert store.aggregate_monitoring_metrics(metrics_path) == {
        "request_latency_ms": {
            "count": 1,
            "total": 50.0,
            "average": 50.0,
        },
        "llm_tokens": {
            "count": 1,
            "total": 25.0,
            "average": 25.0,
        },
    }

    store.clear_monitoring_metrics(metrics_path)

    assert store.read_monitoring_metrics(metrics_path) == []
