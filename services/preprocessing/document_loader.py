from pathlib import Path

class DocumentLoader:

    def load(self, file_path: str):

        path = Path(file_path)

        with open(path, "rb") as f:
            return f.read()