import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Store durable uploaded artifacts locally or in Azure Blob Storage."""

    def put_file(self, local_path: str | Path, object_name: str) -> str:
        backend = settings.ARTIFACT_STORAGE_BACKEND.lower()
        if backend == "local":
            return str(local_path)
        if backend == "azure_blob":
            return self._put_azure_blob_file(local_path, object_name)
        raise RuntimeError(
            "Unsupported ARTIFACT_STORAGE_BACKEND. Use 'local' or 'azure_blob'."
        )

    def get_file(self, storage_uri: str | None, destination_path: str | Path) -> Path:
        destination = Path(destination_path)
        if not storage_uri or not storage_uri.startswith("azure-blob://"):
            return destination

        container, object_name = self._parse_azure_blob_uri(storage_uri)
        blob_client = self._azure_blob_client(container, object_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("wb") as file:
                file.write(blob_client.download_blob().readall())
        except Exception:
            destination.unlink(missing_ok=True)
            logger.exception(
                "failed to download artifact object=%s container=%s",
                object_name,
                container,
            )
            raise
        return destination

    def delete_file(self, storage_uri: str | None) -> None:
        if not storage_uri or not storage_uri.startswith("azure-blob://"):
            return

        container, object_name = self._parse_azure_blob_uri(storage_uri)
        blob_client = self._azure_blob_client(container, object_name)
        try:
            blob_client.delete_blob()
        except Exception:
            logger.exception(
                "failed to delete artifact object=%s container=%s",
                object_name,
                container,
            )
            raise

    def _put_azure_blob_file(self, local_path: str | Path, object_name: str) -> str:
        container_client = self._azure_container_client(settings.AZURE_BLOB_CONTAINER)
        try:
            container_client.create_container()
        except self._azure_resource_exists_error():
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

    def _azure_blob_client(self, container: str, object_name: str):
        return self._azure_container_client(container).get_blob_client(object_name)

    @staticmethod
    def _azure_container_client(container: str):
        try:
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
        return blob_service.get_container_client(container)

    @staticmethod
    def _azure_resource_exists_error():
        try:
            from azure.core.exceptions import ResourceExistsError
        except ImportError as exc:
            raise RuntimeError(
                "The 'azure-storage-blob' package is required for Azure Blob artifact storage."
            ) from exc
        return ResourceExistsError

    @staticmethod
    def _parse_azure_blob_uri(storage_uri: str) -> tuple[str, str]:
        prefix = "azure-blob://"
        if not storage_uri.startswith(prefix):
            raise ValueError(f"Unsupported storage URI: {storage_uri}")

        container_and_object = storage_uri[len(prefix) :]
        container, separator, object_name = container_and_object.partition("/")
        if not container or not separator or not object_name:
            raise ValueError(f"Invalid Azure Blob storage URI: {storage_uri}")
        return container, object_name


artifact_store = ArtifactStore()
