from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Store durable artifacts locally or in MinIO/S3-compatible storage."""

    def put_file(self, local_path: str | Path, object_name: str) -> str | None:
        if settings.ARTIFACT_STORAGE_BACKEND.lower() != "minio":
            return None
        try:
            from minio import Minio
            from minio.error import S3Error
        except ImportError as exc:
            raise RuntimeError("The 'minio' package is required for MinIO artifact storage.") from exc

        if not settings.MINIO_ACCESS_KEY or not settings.MINIO_SECRET_KEY:
            raise RuntimeError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required for MinIO artifact storage.")

        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        try:
            if not client.bucket_exists(settings.MINIO_BUCKET):
                client.make_bucket(settings.MINIO_BUCKET)
            client.fput_object(settings.MINIO_BUCKET, object_name, str(local_path))
        except S3Error:
            logger.exception("failed to store artifact object=%s bucket=%s", object_name, settings.MINIO_BUCKET)
            raise
        return f"minio://{settings.MINIO_BUCKET}/{object_name}"


artifact_store = ArtifactStore()
