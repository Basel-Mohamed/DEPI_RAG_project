from pathlib import Path

class DocumentLoader:

    def load(self, file_path: str) -> bytes:
        path = Path(file_path)
        return path.read_bytes()
