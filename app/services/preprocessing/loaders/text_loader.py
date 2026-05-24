from pathlib import Path
from typing import Any

from app.services.preprocessing.loaders.base_loader import BaseLoader


class TextLoader(BaseLoader):
    def load(self, file_path: Path) -> list[dict[str, Any]]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return [{"text": text, "page_number": 1, "metadata": {}}]
