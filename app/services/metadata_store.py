from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Protocol

from app.core.config import settings


class MetadataStore(Protocol):
    def read_file_registry(self, registry_path: Path) -> dict[str, dict[str, Any]]:
        ...

    def write_file_registry(
        self,
        registry: dict[str, dict[str, Any]],
        registry_path: Path,
    ) -> None:
        ...

    def read_feedback(self, feedback_path: Path) -> list[dict[str, Any]]:
        ...

    def append_feedback(self, record: dict[str, Any], feedback_path: Path) -> None:
        ...


class JsonMetadataStore:
    def read_file_registry(self, registry_path: Path) -> dict[str, dict[str, Any]]:
        if not registry_path.exists():
            return {}
        with registry_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("File registry store is corrupted.")
        return data

    def write_file_registry(
        self,
        registry: dict[str, dict[str, Any]],
        registry_path: Path,
    ) -> None:
        self._write_json(registry_path, registry)

    def read_feedback(self, feedback_path: Path) -> list[dict[str, Any]]:
        if not feedback_path.exists():
            return []
        with feedback_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Feedback store is corrupted.")
        return data

    def append_feedback(self, record: dict[str, Any], feedback_path: Path) -> None:
        feedback = self.read_feedback(feedback_path)
        feedback.append(record)
        self._write_json(feedback_path, feedback)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        temp_path.replace(path)


class SqlMetadataStoreBase:
    file_registry_table: str
    feedback_table: str

    def read_file_registry(self, registry_path: Path) -> dict[str, dict[str, Any]]:
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            rows = connection.execute(
                f"SELECT file_id, payload FROM {self.file_registry_table}"
            ).fetchall()
            return self._registry_from_rows(rows)
        finally:
            connection.close()

    def write_file_registry(
        self,
        registry: dict[str, dict[str, Any]],
        registry_path: Path,
    ) -> None:
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            cursor = connection.cursor()
            cursor.execute(f"DELETE FROM {self.file_registry_table}")
            cursor.executemany(
                f"""
                INSERT INTO {self.file_registry_table} (file_id, payload, updated_at)
                VALUES (?, ?, ?)
                """,
                self._registry_params(registry),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def read_feedback(self, feedback_path: Path) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            rows = connection.execute(
                f"""
                SELECT payload
                FROM {self.feedback_table}
                ORDER BY submitted_at DESC, created_at DESC
                """
            ).fetchall()
            return self._payloads_from_rows(rows)
        finally:
            connection.close()

    def append_feedback(self, record: dict[str, Any], feedback_path: Path) -> None:
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            connection.execute(
                f"""
                INSERT INTO {self.feedback_table} (
                    feedback_id, session_id, rating, submitted_at, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                self._feedback_params(record),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self):
        raise NotImplementedError

    def _ensure_schema(self, connection) -> None:
        raise NotImplementedError

    @staticmethod
    def _registry_from_rows(rows) -> dict[str, dict[str, Any]]:
        return {file_id: json.loads(payload) for file_id, payload in rows}

    @staticmethod
    def _payloads_from_rows(rows) -> list[dict[str, Any]]:
        return [json.loads(payload) for (payload,) in rows]

    @staticmethod
    def _registry_params(registry: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
        return [
            (
                file_id,
                json.dumps(payload, sort_keys=True),
                payload.get("updated_at") or payload.get("created_at") or "",
            )
            for file_id, payload in registry.items()
        ]

    @staticmethod
    def _feedback_params(record: dict[str, Any]) -> tuple[str, str, int, str, str, str]:
        return (
            record["feedback_id"],
            record["session_id"],
            record["rating"],
            record["timestamp"],
            record["created_at"],
            json.dumps(record, sort_keys=True),
        )


class SqliteMetadataStore(SqlMetadataStoreBase):
    file_registry_table = "file_registry"
    feedback_table = "feedback"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.METADATA_DB_PATH)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS file_registry (
                file_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                submitted_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback(session_id)"
        )


class AzureSqlMetadataStore(SqlMetadataStoreBase):
    file_registry_table = "dbo.file_registry"
    feedback_table = "dbo.feedback"

    @staticmethod
    def _connect():
        if not settings.AZURE_SQL_CONNECTION_STRING:
            raise RuntimeError(
                "AZURE_SQL_CONNECTION_STRING is required for Azure SQL metadata storage."
            )
        try:
            import pyodbc
        except ImportError as exc:
            raise RuntimeError(
                "The 'pyodbc' package is required for Azure SQL metadata storage."
            ) from exc
        try:
            return pyodbc.connect(settings.AZURE_SQL_CONNECTION_STRING)
        except pyodbc.InterfaceError as exc:
            if "IM002" not in str(exc):
                raise
            drivers = ", ".join(pyodbc.drivers()) or "none"
            raise RuntimeError(
                "Azure SQL connection failed because the configured ODBC driver "
                "was not found. Install 'ODBC Driver 18 for SQL Server' or update "
                f"AZURE_SQL_CONNECTION_STRING to use an installed driver. "
                f"Installed ODBC drivers: {drivers}."
            ) from exc

    @staticmethod
    def _ensure_schema(connection) -> None:
        connection.execute(
            """
            IF OBJECT_ID(N'dbo.file_registry', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.file_registry (
                    file_id NVARCHAR(100) NOT NULL PRIMARY KEY,
                    payload NVARCHAR(MAX) NOT NULL,
                    updated_at NVARCHAR(50) NOT NULL
                )
            END
            """
        )
        connection.execute(
            """
            IF OBJECT_ID(N'dbo.feedback', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.feedback (
                    feedback_id NVARCHAR(100) NOT NULL PRIMARY KEY,
                    session_id NVARCHAR(255) NOT NULL,
                    rating INT NOT NULL,
                    submitted_at NVARCHAR(50) NOT NULL,
                    created_at NVARCHAR(50) NOT NULL,
                    payload NVARCHAR(MAX) NOT NULL
                )
            END
            """
        )
        connection.execute(
            """
            IF NOT EXISTS (
                SELECT 1
                FROM sys.indexes
                WHERE name = N'idx_feedback_session_id'
                  AND object_id = OBJECT_ID(N'dbo.feedback')
            )
            BEGIN
                CREATE INDEX idx_feedback_session_id ON dbo.feedback(session_id)
            END
            """
        )
        connection.commit()


def get_metadata_store() -> MetadataStore:
    backend = settings.METADATA_BACKEND.lower()
    if backend == "json":
        return JsonMetadataStore()
    if backend == "sqlite":
        return SqliteMetadataStore()
    if backend == "azure_sql":
        return AzureSqlMetadataStore()
    raise ValueError("Unsupported METADATA_BACKEND. Use 'json', 'sqlite', or 'azure_sql'.")
