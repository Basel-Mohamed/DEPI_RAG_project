import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.dependencies import get_build_service
from app.controllers.build_controller import BuildController
from app.services.rag.rag_builder import BuildService
from main import app


class FakeDocumentProcessor:
    def __init__(self) -> None:
        self.seen_paths: list[Path] = []
        self.chunks = [
            {
                "text": "Build pipelines turn PDFs into searchable chunks.",
                "metadata": {"source": "temporary-path.pdf", "page_number": 1, "chunk_index": 0},
            },
            {
                "text": "Cleanup keeps test vector stores isolated.",
                "metadata": {"source": "temporary-path.pdf", "page_number": 1, "chunk_index": 1},
            },
        ]

    def process_document(self, file_path: str | Path) -> list[dict]:
        path = Path(file_path)
        assert path.exists()
        self.seen_paths.append(path)
        return [{**chunk, "metadata": dict(chunk["metadata"])} for chunk in self.chunks]


class FakeEmbeddingService:
    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        return [{**chunk, "embedding": [0.1, 0.2, 0.3]} for chunk in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: list[dict] = []

    def upsert(self, embedded_chunks: list[dict]) -> dict:
        by_id = {
            self._point_key(point): point
            for point in self.points
        }
        for chunk in embedded_chunks:
            by_id[self._point_key(chunk)] = chunk
        self.points = list(by_id.values())
        return {"upserted": len(embedded_chunks), "failed": 0}

    def scroll(self, filter_field: str | None = None, filter_value=None, limit: int = 100) -> list[dict]:
        if filter_field is None:
            return self.points[:limit]
        return [
            point
            for point in self.points
            if point["metadata"].get(filter_field) == filter_value
        ][:limit]

    def count_by_filter(self, filter_field: str, filter_value) -> int:
        return len(self.scroll(filter_field=filter_field, filter_value=filter_value, limit=10_000))

    def delete_by_filter(self, filter_field: str, filter_value, exclude: dict | None = None) -> dict:
        before = len(self.points)
        self.points = [
            point
            for point in self.points
            if point["metadata"].get(filter_field) != filter_value
            or self._matches_exclude(point, exclude)
        ]
        return {"deleted_count": before - len(self.points)}

    @staticmethod
    def _point_key(point: dict) -> tuple:
        metadata = point["metadata"]
        return metadata["source"], metadata["page_number"], metadata["chunk_index"]

    @staticmethod
    def _matches_exclude(point: dict, exclude: dict | None) -> bool:
        return bool(exclude) and all(
            point["metadata"].get(key) == value
            for key, value in exclude.items()
        )


class FakeArtifactStore:
    def __init__(self) -> None:
        self.puts: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.deletes: list[str | None] = []

    def put_file(self, local_path: str | Path, object_name: str) -> str:
        self.puts.append((Path(local_path), object_name))
        return f"azure-blob://test-container/{object_name}"

    def get_file(self, storage_uri: str | None, destination_path: str | Path) -> Path:
        destination = Path(destination_path)
        self.downloads.append((storage_uri or "", destination))
        destination.write_bytes(b"%PDF-1.4 restored pdf bytes")
        return destination

    def delete_file(self, storage_uri: str | None) -> None:
        self.deletes.append(storage_uri)


@pytest.fixture
def fake_build_stack():
    processor = FakeDocumentProcessor()
    vector_store = FakeVectorStore()
    service = BuildService(
        document_processor=processor,
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )
    return service, processor, vector_store


@pytest.fixture
def client(fake_build_stack, tmp_path, monkeypatch):
    service, _, _ = fake_build_stack
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(settings, "ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.setattr(BuildController, "upload_root", tmp_path / "uploads")
    monkeypatch.setattr(BuildController, "registry_path", tmp_path / "uploads" / "files.json")
    app.dependency_overrides[get_build_service] = lambda: service
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_file_upload_rejects_unsupported_upload(client):
    response = client.post(
        "/files",
        files={"file": ("notes.exe", b"not supported", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_file_upload_rejects_duplicate_filename(client):
    first_response = client.post(
        "/files",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    second_response = client.post(
        "/files",
        files={"file": ("sample.pdf", b"%PDF-1.4 different fake bytes", "application/pdf")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert "Duplicate document filename already uploaded" in second_response.json()["detail"]


def test_file_upload_rejects_duplicate_content(client):
    first_response = client.post(
        "/files",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    second_response = client.post(
        "/files",
        files={"file": ("renamed.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert "Duplicate document content already uploaded" in second_response.json()["detail"]


def test_protected_build_endpoint_rejects_missing_api_key(fake_build_stack, tmp_path, monkeypatch):
    service, _, _ = fake_build_stack
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(settings, "ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.setattr(BuildController, "upload_root", tmp_path / "uploads")
    monkeypatch.setattr(BuildController, "registry_path", tmp_path / "uploads" / "files.json")
    app.dependency_overrides[get_build_service] = lambda: service
    with TestClient(app) as test_client:
        response = test_client.get("/files")
    app.dependency_overrides.clear()

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_protected_build_endpoint_rejects_invalid_api_key(fake_build_stack, tmp_path, monkeypatch):
    service, _, _ = fake_build_stack
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(settings, "ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.setattr(BuildController, "upload_root", tmp_path / "uploads")
    monkeypatch.setattr(BuildController, "registry_path", tmp_path / "uploads" / "files.json")
    app.dependency_overrides[get_build_service] = lambda: service
    with TestClient(app) as test_client:
        response = test_client.get("/files", headers={"X-API-Key": "wrong-key"})
    app.dependency_overrides.clear()

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_file_lifecycle_endpoints_upload_build_list_get_and_delete(client, fake_build_stack):
    _, _, vector_store = fake_build_stack

    upload_response = client.post(
        "/files",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    assert upload_response.status_code == 201
    uploaded = upload_response.json()
    file_id = uploaded["file_id"]
    assert uploaded == {
        "file_id": file_id,
        "filename": "sample.pdf",
        "content_type": "application/pdf",
        "status": "uploaded",
    }

    build_response = client.post(f"/files/build?file_id={file_id}")
    assert build_response.status_code == 200
    assert build_response.json()["files"] == [
        {
            "file_id": file_id,
            "filename": "sample.pdf",
            "content_type": "application/pdf",
            "status": "built",
            "chunks_count": 2,
            "upserted": 2,
            "failed": 0,
            "last_error": None,
        }
    ]
    assert {point["metadata"]["source"] for point in vector_store.points} == {file_id}

    get_response = client.get(f"/files?file_id={file_id}")
    assert get_response.status_code == 200
    assert get_response.json()["status"] == "built"
    assert get_response.json()["chunks_count"] == 2

    list_response = client.get("/files")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
    assert list_response.json()[0]["file_id"] == file_id

    delete_response = client.delete(f"/files?file_id={file_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "file_id": file_id,
        "deleted_count": 2,
        "files_deleted": 1,
    }
    assert vector_store.points == []


def test_file_upload_persists_pdf_to_artifact_store(fake_build_stack, tmp_path, monkeypatch):
    service, _, _ = fake_build_stack
    fake_store = FakeArtifactStore()
    registry_path = tmp_path / "uploads" / "files.json"
    monkeypatch.setattr(settings, "API_KEY", "test-api-key")
    monkeypatch.setattr(settings, "METADATA_BACKEND", "json")
    monkeypatch.setattr(BuildController, "upload_root", tmp_path / "uploads")
    monkeypatch.setattr(BuildController, "registry_path", registry_path)
    monkeypatch.setattr(BuildController, "artifact_store", fake_store)
    app.dependency_overrides[get_build_service] = lambda: service

    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as test_client:
        upload_response = test_client.post(
            "/files",
            files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
        )
        assert upload_response.status_code == 201
        file_id = upload_response.json()["file_id"]
        cached_path = tmp_path / "uploads" / f"{file_id}.pdf"
        assert not cached_path.exists()

        build_response = test_client.post(f"/files/build?file_id={file_id}")
    app.dependency_overrides.clear()

    assert fake_store.puts == [
        (tmp_path / "uploads" / f"{file_id}.pdf", f"uploads/{file_id}.pdf")
    ]
    assert build_response.status_code == 200
    assert not cached_path.exists()
    assert fake_store.downloads == [
        (
            f"azure-blob://test-container/uploads/{file_id}.pdf",
            tmp_path / "uploads" / f"{file_id}.pdf",
        )
    ]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry[file_id]["storage_uri"] == (
        f"azure-blob://test-container/uploads/{file_id}.pdf"
    )


def test_build_status_counts_more_than_default_scroll_limit(client, fake_build_stack):
    _, processor, _ = fake_build_stack
    processor.chunks = [
        {
            "text": f"Chunk {index}",
            "metadata": {"source": "temporary-path.pdf", "page_number": 1, "chunk_index": index},
        }
        for index in range(125)
    ]

    upload_response = client.post(
        "/files",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    file_id = upload_response.json()["file_id"]
    build_response = client.post(
        f"/files/build?file_id={file_id}",
    )
    assert build_response.status_code == 200

    status_response = client.get(f"/files?file_id={file_id}")
    assert status_response.status_code == 200
    assert status_response.json()["chunks_count"] == 125


def test_rebuild_removes_stale_chunks_for_same_source(client, fake_build_stack):
    _, processor, vector_store = fake_build_stack

    upload_response = client.post(
        "/files",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    file_id = upload_response.json()["file_id"]

    first_response = client.post(f"/files/build?file_id={file_id}")
    assert first_response.status_code == 200
    assert len(vector_store.scroll("source", file_id, limit=10)) == 2

    processor.chunks = [
        {
            "text": "Only one chunk after rebuild.",
            "metadata": {"source": "temporary-path.pdf", "page_number": 1, "chunk_index": 0},
        }
    ]
    second_response = client.post(f"/files/build?file_id={file_id}")
    assert second_response.status_code == 200

    source_points = vector_store.scroll("source", file_id, limit=10)
    assert len(source_points) == 1
    assert source_points[0]["text"] == "Only one chunk after rebuild."


def test_legacy_direct_build_routes_are_not_available(client):
    document_response = client.post(
        "/documents",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    upload_response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )

    assert document_response.status_code == 404
    assert upload_response.status_code == 404


def test_path_parameter_file_lookup_route_is_not_available(client):
    response = client.get("/files/not-a-route")

    assert response.status_code == 404
