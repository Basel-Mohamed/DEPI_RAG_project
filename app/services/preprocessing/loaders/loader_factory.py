from pathlib import Path

from app.services.preprocessing.loaders.base_loader import BaseLoader
from app.services.preprocessing.loaders.csv_loader import CsvLoader
from app.services.preprocessing.loaders.json_loader import JsonLoader
from app.services.preprocessing.loaders.text_loader import TextLoader

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json", ".txt", ".md"}


def get_loader(file_path: Path) -> BaseLoader:
    """Return the correct loader for the file extension."""

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from app.services.preprocessing.loaders.pdf_loader import PdfLoader

        return PdfLoader()
    if suffix == ".csv":
        return CsvLoader()
    if suffix == ".json":
        return JsonLoader()
    if suffix in {".txt", ".md"}:
        return TextLoader()

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported file extension '{suffix}'. Supported extensions: {supported}.")
