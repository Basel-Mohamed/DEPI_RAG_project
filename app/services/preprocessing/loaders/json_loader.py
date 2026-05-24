import json
from pathlib import Path
from typing import Any

from app.services.preprocessing.loaders.base_loader import BaseLoader


class JsonLoader(BaseLoader):
    def __init__(self, text_key: str = "answer", title_key: str = "question") -> None:
        self.text_key = text_key
        self.title_key = title_key

    def load(self, file_path: Path) -> list[dict[str, Any]]:
        with file_path.open("r", encoding="utf-8", errors="replace") as file:
            data = json.load(file)

        items = data if isinstance(data, list) else [data]
        pages: list[dict[str, Any]] = []

        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                item = {self.text_key: str(item)}

            text = str(item.get(self.text_key, ""))
            title = item.get(self.title_key)
            if title is not None and str(title).strip():
                text = f"{title}\n{text}"

            metadata = {
                key: value
                for key, value in item.items()
                if key not in {self.text_key, self.title_key}
            }
            pages.append({
                "text": text,
                "page_number": item_index + 1,
                "metadata": metadata,
            })

        return pages
