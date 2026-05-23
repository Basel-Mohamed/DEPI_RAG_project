from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.core.config import settings
from app.services.artifacts import ArtifactStore


def test_local_artifact_backend_does_not_mirror_file(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(settings, "ARTIFACT_STORAGE_BACKEND", "local")

    assert ArtifactStore().put_file(file_path, "uploads/sample.pdf") is None


def test_azure_blob_artifact_backend_uploads_file(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(settings, "ARTIFACT_STORAGE_BACKEND", "azure_blob")
    monkeypatch.setattr(settings, "AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    monkeypatch.setattr(settings, "AZURE_BLOB_CONTAINER", "rag-artifacts")
    captured: dict[str, object] = {}

    class ResourceExistsError(Exception):
        pass

    class FakeBlobClient:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_blob(self, file, overwrite: bool) -> None:
            captured["blob_name"] = self.name
            captured["content"] = file.read()
            captured["overwrite"] = overwrite

    class FakeContainerClient:
        def __init__(self, name: str) -> None:
            self.name = name

        def create_container(self) -> None:
            captured["container_created"] = self.name

        def get_blob_client(self, object_name: str) -> FakeBlobClient:
            return FakeBlobClient(object_name)

    class FakeBlobServiceClient:
        @classmethod
        def from_connection_string(cls, connection_string: str) -> "FakeBlobServiceClient":
            captured["connection_string"] = connection_string
            return cls()

        def get_container_client(self, container_name: str) -> FakeContainerClient:
            captured["container_name"] = container_name
            return FakeContainerClient(container_name)

    _install_fake_azure_blob_modules(
        monkeypatch,
        blob_service_client=FakeBlobServiceClient,
        resource_exists_error=ResourceExistsError,
    )

    uri = ArtifactStore().put_file(file_path, "uploads/sample.pdf")

    assert uri == "azure-blob://rag-artifacts/uploads/sample.pdf"
    assert captured == {
        "connection_string": "UseDevelopmentStorage=true",
        "container_name": "rag-artifacts",
        "container_created": "rag-artifacts",
        "blob_name": "uploads/sample.pdf",
        "content": b"%PDF-1.4",
        "overwrite": True,
    }


def test_azure_blob_artifact_backend_requires_connection_string(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ARTIFACT_STORAGE_BACKEND", "azure_blob")
    monkeypatch.setattr(settings, "AZURE_STORAGE_CONNECTION_STRING", None)
    _install_fake_azure_blob_modules(monkeypatch)

    with pytest.raises(RuntimeError, match="AZURE_STORAGE_CONNECTION_STRING"):
        ArtifactStore().put_file("missing.pdf", "uploads/missing.pdf")


def _install_fake_azure_blob_modules(
    monkeypatch,
    blob_service_client: type | None = None,
    resource_exists_error: type[Exception] | None = None,
) -> None:
    azure_module = types.ModuleType("azure")
    core_module = types.ModuleType("azure.core")
    exceptions_module = types.ModuleType("azure.core.exceptions")
    storage_module = types.ModuleType("azure.storage")
    blob_module = types.ModuleType("azure.storage.blob")

    exceptions_module.ResourceExistsError = resource_exists_error or type(
        "ResourceExistsError",
        (Exception,),
        {},
    )
    blob_module.BlobServiceClient = blob_service_client or type("BlobServiceClient", (), {})

    monkeypatch.setitem(sys.modules, "azure", azure_module)
    monkeypatch.setitem(sys.modules, "azure.core", core_module)
    monkeypatch.setitem(sys.modules, "azure.core.exceptions", exceptions_module)
    monkeypatch.setitem(sys.modules, "azure.storage", storage_module)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", blob_module)
