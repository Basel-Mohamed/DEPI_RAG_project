import csv
from pathlib import Path
from typing import Any

from app.services.preprocessing.loaders.base_loader import BaseLoader


class CsvLoader(BaseLoader):
    def __init__(self, text_columns: list[str] | None = None) -> None:
        self.text_columns = text_columns or [
            "subject",
            "body",
            "description",
            "text",
            "content",
            "message",
        ]

    def load(self, file_path: Path) -> list[dict[str, Any]]:
        rows = self._read_rows(file_path, delimiter=",")
        if not rows:
            rows = self._read_rows(file_path, delimiter=";")

        present_text_columns = [
            column for column in self.text_columns if rows and column in rows[0]
        ]
        pages: list[dict[str, Any]] = []

        for row_index, row in enumerate(rows):
            text_parts = [
                str(row[column])
                for column in present_text_columns
                if row.get(column) is not None and str(row[column]).strip()
            ]
            metadata = {
                key: value
                for key, value in row.items()
                if key not in present_text_columns
            }
            pages.append({
                "text": " ".join(text_parts),
                "page_number": row_index + 1,
                "metadata": metadata,
            })

        return pages

    def _read_rows(self, file_path: Path, delimiter: str) -> list[dict[str, Any]]:
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            rows = list(reader)
            if reader.fieldnames is None:
                return []
            if delimiter == "," and len(reader.fieldnames) == 1:
                return []
            return rows
