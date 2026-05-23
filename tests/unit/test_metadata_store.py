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
