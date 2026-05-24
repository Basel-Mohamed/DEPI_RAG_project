import hashlib
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.services.artifacts import artifact_store
from app.services.metadata_store import get_metadata_store
from app.services.preprocessing.loaders.loader_factory import SUPPORTED_EXTENSIONS
from app.services.rag.rag_builder import BuildService

logger = logging.getLogger(__name__)


class BuildController:
    upload_root = Path.cwd() / "uploads"
    registry_path = upload_root / "files.json"
    registry_lock = threading.RLock()
    max_upload_bytes = 50 * 1024 * 1024
    artifact_store = artifact_store

    def __init__(self, build_service: BuildService) -> None:
        self.build_service = build_service

    async def upload_file(self, file: UploadFile) -> dict:
        if not file.filename:
            raise ValueError("A file name is required.")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Only these document formats are supported: {supported}.")

        with self.registry_lock:
            registry = self._read_registry()
            duplicate = self._find_duplicate_filename(registry, file.filename)
            if duplicate is not None:
                raise ValueError(
                    f"Duplicate document filename already uploaded: {file.filename} "
                    f"(file_id={duplicate['file_id']})."
                )

        self.upload_root.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        stored_path = self.upload_root / f"{file_id}{suffix}"

        logger.info("file upload save started file_id=%s filename=%s", file_id, file.filename)
        file_sha256 = self._save_upload(file, stored_path)
        if suffix == ".pdf":
            self._validate_pdf_file(stored_path)
        with self.registry_lock:
            registry = self._read_registry()
            duplicate = self._find_duplicate_hash(registry, file_sha256)
            if duplicate is not None:
                stored_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Duplicate document content already uploaded as "
                    f"{duplicate['filename']} (file_id={duplicate['file_id']})."
                )

        object_name = f"uploads/{file_id}{suffix}"
        storage_uri = self.artifact_store.put_file(stored_path, object_name)
        if self._is_remote_storage_uri(storage_uri):
            stored_path.unlink(missing_ok=True)

        now = self._utc_now()
        metadata = {
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "sha256": file_sha256,
            "path": str(stored_path),
            "storage_uri": storage_uri,
            "status": "uploaded",
            "chunks_count": 0,
            "upserted": 0,
            "failed": 0,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
        }
        with self.registry_lock:
            registry = self._read_registry()
            registry[file_id] = metadata
            self._write_registry(registry)
        logger.info("file upload save completed file_id=%s path=%s", file_id, stored_path)
        return self._public_file(metadata)

    def build_files(self, file_id: str | None = None) -> dict:
        with self.registry_lock:
            registry = self._read_registry()
            files = [self._get_file_or_raise(registry, file_id)] if file_id else list(registry.values())

        results = []
        for metadata in files:
            path = Path(metadata["path"])
            restored_from_remote = False
            if not path.exists():
                storage_uri = metadata.get("storage_uri")
                if storage_uri:
                    try:
                        self.artifact_store.get_file(storage_uri, path)
                        restored_from_remote = self._is_remote_storage_uri(storage_uri)
                    except Exception:
                        logger.exception(
                            "file cache restore failed file_id=%s storage_uri=%s",
                            metadata["file_id"],
                            storage_uri,
                        )

            if not path.exists():
                metadata["status"] = "missing"
                metadata["updated_at"] = self._utc_now()
                results.append(self._public_file(metadata))
                with self.registry_lock:
                    registry[metadata["file_id"]] = metadata
                    self._write_registry(registry)
                continue

            metadata["status"] = "building"
            metadata["last_error"] = None
            metadata["updated_at"] = self._utc_now()
            with self.registry_lock:
                registry[metadata["file_id"]] = metadata
                self._write_registry(registry)
            try:
                build_result = self.build_service.build_document(path, source=metadata["file_id"])
            except Exception as exc:
                metadata["status"] = "failed"
                metadata["last_error"] = str(exc)
                metadata["updated_at"] = self._utc_now()
                with self.registry_lock:
                    registry[metadata["file_id"]] = metadata
                    self._write_registry(registry)
                raise
            finally:
                if restored_from_remote:
                    path.unlink(missing_ok=True)

            metadata.update(
                {
                    "status": "built",
                    "chunks_count": build_result["chunks_count"],
                    "upserted": build_result["upserted"],
                    "failed": build_result["failed"],
                    "last_error": None,
                    "updated_at": self._utc_now(),
                }
            )
            results.append(self._public_file(metadata))

        with self.registry_lock:
            registry = self._read_registry()
            for metadata in files:
                registry[metadata["file_id"]] = metadata
            self._write_registry(registry)
        return {"files": results}

    def get_file(self, file_id: str) -> dict:
        with self.registry_lock:
            registry = self._read_registry()
            metadata = self._get_file_or_raise(registry, file_id)
        if metadata["status"] in {"built", "building"}:
            status = self.build_service.get_document_status(file_id)
            metadata["chunks_count"] = status["chunks_count"]
            metadata["updated_at"] = self._utc_now()
            with self.registry_lock:
                registry[file_id] = metadata
                self._write_registry(registry)
        return self._public_file(metadata)

    def list_files(self) -> list[dict]:
        with self.registry_lock:
            registry = self._read_registry()
        changed = False
        files = []
        for metadata in registry.values():
            if metadata["status"] in {"built", "building"}:
                status = self.build_service.get_document_status(metadata["file_id"])
                metadata["chunks_count"] = status["chunks_count"]
                metadata["updated_at"] = self._utc_now()
                changed = True
            files.append(self._public_file(metadata))
        if changed:
            with self.registry_lock:
                self._write_registry(registry)
        return files

    def delete_files(self, file_id: str | None = None) -> dict:
        with self.registry_lock:
            registry = self._read_registry()
        target_ids = [file_id] if file_id else list(registry)
        deleted_count = 0
        files_deleted = 0

        for target_id in target_ids:
            if target_id is None:
                continue
            metadata = self._get_file_or_raise(registry, target_id)
            deleted_count += self.build_service.delete_document(target_id)["deleted_count"]
            self.artifact_store.delete_file(metadata.get("storage_uri"))
            Path(metadata["path"]).unlink(missing_ok=True)
            registry.pop(target_id, None)
            files_deleted += 1

        with self.registry_lock:
            self._write_registry(registry)
        return {
            "file_id": file_id,
            "deleted_count": deleted_count,
            "files_deleted": files_deleted,
        }

    def _read_registry(self) -> dict[str, dict[str, Any]]:
        return get_metadata_store().read_file_registry(self.registry_path)

    def _write_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        get_metadata_store().write_file_registry(registry, self.registry_path)

    @staticmethod
    def _get_file_or_raise(registry: dict[str, dict[str, Any]], file_id: str | None) -> dict[str, Any]:
        if file_id and file_id in registry:
            return registry[file_id]
        raise FileNotFoundError(f"File not found: {file_id}")

    @staticmethod
    def _public_file(metadata: dict[str, Any]) -> dict:
        return {
            "file_id": metadata["file_id"],
            "filename": metadata["filename"],
            "content_type": metadata.get("content_type"),
            "status": metadata["status"],
            "chunks_count": metadata.get("chunks_count", 0),
            "upserted": metadata.get("upserted", 0),
            "failed": metadata.get("failed", 0),
            "last_error": metadata.get("last_error"),
        }

    @staticmethod
    def _save_upload(file: UploadFile, path: Path) -> str:
        total_bytes = 0
        hasher = hashlib.sha256()
        try:
            with path.open("wb") as buffer:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > BuildController.max_upload_bytes:
                        raise ValueError("Document is too large.")
                    hasher.update(chunk)
                    buffer.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return hasher.hexdigest()

    @staticmethod
    def _find_duplicate_filename(
        registry: dict[str, dict[str, Any]],
        filename: str,
    ) -> dict[str, Any] | None:
        normalized_filename = filename.casefold()
        for metadata in registry.values():
            if str(metadata.get("filename", "")).casefold() == normalized_filename:
                return metadata
        return None

    @classmethod
    def _find_duplicate_hash(
        cls,
        registry: dict[str, dict[str, Any]],
        file_sha256: str,
    ) -> dict[str, Any] | None:
        for metadata in registry.values():
            existing_hash = metadata.get("sha256")
            if existing_hash is None:
                existing_hash = cls._hash_existing_local_file(metadata)
            if existing_hash == file_sha256:
                return metadata
        return None

    @staticmethod
    def _hash_existing_local_file(metadata: dict[str, Any]) -> str | None:
        path = Path(str(metadata.get("path", "")))
        if not path.exists() or not path.is_file():
            return None

        hasher = hashlib.sha256()
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _validate_pdf_file(path: Path) -> None:
        with path.open("rb") as file:
            signature = file.read(5)
        if signature != b"%PDF-":
            path.unlink(missing_ok=True)
            raise ValueError("Only valid PDF documents are supported.")

    @staticmethod
    def _is_remote_storage_uri(storage_uri: str | None) -> bool:
        return bool(storage_uri and storage_uri.startswith("azure-blob://"))

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()
