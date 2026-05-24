from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> list[dict[str, Any]]:
        """Return list of page dicts: {"text": str, "page_number": int, "metadata": dict}"""
