import logging
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.services.rag.rag_builder import BuildService

logger = logging.getLogger(__name__)


class BuildController:
    upload_root = Path.cwd() / "uploads"
    registry_path = upload_root / "files.json"
    registry_lock = threading.RLock()
    max_upload_bytes = 50 * 1024 * 1024

    def __init__(self, build_service: BuildService) -> None:
        self.build_service = build_service

    async def upload_file(self, file: UploadFile) -> dict:
        if not file.filename:
            raise ValueError("A file name is required.")
        suffix = Path(file.filename).suffix.lower()
        if suffix != ".pdf":
            raise ValueError("Only PDF documents are supported.")

        self.upload_root.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        stored_path = self.upload_root / f"{file_id}{suffix}"

        logger.info("file upload save started file_id=%s filename=%s", file_id, file.filename)
        self._save_upload(file, stored_path)
        self._validate_pdf_file(stored_path)

        now = self._utc_now()
        metadata = {
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "path": str(stored_path),
            "status": "uploaded",
            "chunks_count": 0,
            "page_images_count": 0,
            "chunks_with_page_images_count": 0,
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

            metadata.update(
                {
                    "status": "built",
                    "chunks_count": build_result["chunks_count"],
                    "page_images_count": build_result["page_images_count"],
                    "chunks_with_page_images_count": build_result.get("chunks_with_page_images_count", 0),
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
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        temp_path = self.registry_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(registry, file, indent=2, sort_keys=True)
        os.replace(temp_path, self.registry_path)

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
            "page_images_count": metadata.get("page_images_count", 0),
            "chunks_with_page_images_count": metadata.get("chunks_with_page_images_count", 0),
            "upserted": metadata.get("upserted", 0),
            "failed": metadata.get("failed", 0),
            "last_error": metadata.get("last_error"),
        }

    @staticmethod
    def _save_upload(file: UploadFile, path: Path) -> None:
        total_bytes = 0
        try:
            with path.open("wb") as buffer:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > BuildController.max_upload_bytes:
                        raise ValueError("PDF document is too large.")
                    buffer.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_pdf_file(path: Path) -> None:
        with path.open("rb") as file:
            signature = file.read(5)
        if signature != b"%PDF-":
            path.unlink(missing_ok=True)
            raise ValueError("Only valid PDF documents are supported.")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()
