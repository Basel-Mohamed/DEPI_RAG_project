from app.controllers.build_controller import BuildController
from app.core.config import settings


def test_sqlite_registry_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "METADATA_BACKEND", "sqlite")
    monkeypatch.setattr(settings, "METADATA_DB_PATH", str(tmp_path / "metadata.sqlite3"))
    controller = BuildController()
    payload = {
        "file-1": {
            "file_id": "file-1",
            "filename": "sample.pdf",
            "content_type": "application/pdf",
            "path": "uploads/file-1.pdf",
            "status": "uploaded",
            "chunks_count": 0,
            "upserted": 0,
            "failed": 0,
            "last_error": None,
        }
    }

    controller._write_registry(payload)

    assert controller._read_registry() == payload
