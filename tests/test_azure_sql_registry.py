from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from app.controllers.build_controller import BuildController
from app.core.config import settings


def test_azure_sql_registry_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyodbc = _FakePyodbc()
    monkeypatch.setitem(sys.modules, "pyodbc", fake_pyodbc.module)
    monkeypatch.setattr(settings, "METADATA_BACKEND", "azure_sql")
    monkeypatch.setattr(
        settings,
        "AZURE_SQL_CONNECTION_STRING",
        "Driver={ODBC Driver 18 for SQL Server};Server=tcp:example.database.windows.net,1433;",
    )
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
    assert fake_pyodbc.connection_strings == [
        settings.AZURE_SQL_CONNECTION_STRING,
        settings.AZURE_SQL_CONNECTION_STRING,
    ]


def test_azure_sql_registry_requires_connection_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "METADATA_BACKEND", "azure_sql")
    monkeypatch.setattr(settings, "AZURE_SQL_CONNECTION_STRING", None)

    with pytest.raises(RuntimeError, match="AZURE_SQL_CONNECTION_STRING"):
        BuildController()._read_registry()


class _FakePyodbc:
    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.connection_strings: list[str] = []
        self.module = types.ModuleType("pyodbc")
        self.module.connect = self.connect

    def connect(self, connection_string: str) -> "_FakeConnection":
        self.connection_strings.append(connection_string)
        return _FakeConnection(self.rows)


class _FakeConnection:
    def __init__(self, rows: dict[str, str]) -> None:
        self.rows = rows
        self.closed = False

    def execute(self, sql: str, *params: Any) -> "_FakeCursor":
        return _FakeCursor(self.rows).execute(sql, *params)

    def cursor(self) -> "_FakeCursor":
        return _FakeCursor(self.rows)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeCursor:
    def __init__(self, rows: dict[str, str]) -> None:
        self.rows = rows
        self.result: list[tuple[str, str]] = []

    def execute(self, sql: str, *params: Any) -> "_FakeCursor":
        normalized = " ".join(sql.split()).upper()
        if normalized.startswith("DELETE FROM DBO.FILE_REGISTRY"):
            self.rows.clear()
        elif normalized.startswith("INSERT INTO DBO.FILE_REGISTRY"):
            file_id, payload, _ = params
            self.rows[file_id] = payload
        elif normalized.startswith("SELECT FILE_ID, PAYLOAD FROM DBO.FILE_REGISTRY"):
            self.result = list(self.rows.items())
        return self

    def executemany(self, sql: str, values: list[tuple[str, str, str]]) -> None:
        for params in values:
            self.execute(sql, *params)

    def fetchall(self) -> list[tuple[str, str]]:
        return self.result
