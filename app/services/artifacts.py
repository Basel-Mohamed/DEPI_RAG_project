from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Store durable artifacts locally, in MinIO/S3, or in Azure Blob Storage."""

    def put_file(self, local_path: str | Path, object_name: str) -> str | None:
        backend = settings.ARTIFACT_STORAGE_BACKEND.lower()
        if backend == "local":
            return None
        if backend == "minio":
            return self._put_minio_file(local_path, object_name)
        if backend == "azure_blob":
            return self._put_azure_blob_file(local_path, object_name)
        raise RuntimeError(
            "Unsupported ARTIFACT_STORAGE_BACKEND. Use 'local', 'minio', or 'azure_blob'."
        )

    def _put_minio_file(self, local_path: str | Path, object_name: str) -> str:
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

    def _put_azure_blob_file(self, local_path: str | Path, object_name: str) -> str:
        try:
            from azure.core.exceptions import ResourceExistsError
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise RuntimeError(
                "The 'azure-storage-blob' package is required for Azure Blob artifact storage."
            ) from exc

        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING is required for Azure Blob artifact storage."
            )

        blob_service = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        container_client = blob_service.get_container_client(settings.AZURE_BLOB_CONTAINER)
        try:
            container_client.create_container()
        except ResourceExistsError:
            pass

        try:
            blob_client = container_client.get_blob_client(object_name)
            with Path(local_path).open("rb") as file:
                blob_client.upload_blob(file, overwrite=True)
        except Exception:
            logger.exception(
                "failed to store artifact object=%s container=%s",
                object_name,
                settings.AZURE_BLOB_CONTAINER,
            )
            raise
        return f"azure-blob://{settings.AZURE_BLOB_CONTAINER}/{object_name}"


artifact_store = ArtifactStore()
