import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings as global_settings
from app.services.preprocessing.deduplicator import deduplicate_chunks
from app.services.preprocessing.loaders.loader_factory import SUPPORTED_EXTENSIONS, get_loader
from app.services.preprocessing.pii_cleaner import redact_pii
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Load a supported document, split page text into chunks."""

    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or global_settings

        self.chunk_size = self.settings.CHUNK_SIZE
        self.chunk_overlap = self.settings.CHUNK_OVERLAP
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def process_document(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Run the text-only processing pipeline for a single document."""

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info("Processing document: %s", file_path)

        pages = self._load_pages(file_path)
        chunks = self._split_and_clean(pages, source=str(file_path))
        chunks, removed_count = deduplicate_chunks(chunks)
        logger.info(
            "Deduplication removed %d duplicate chunks from source=%s",
            removed_count,
            file_path,
        )

        logger.info("Produced %d chunks from %d pages.", len(chunks), len(pages))
        return chunks

    def _load_pages(self, file_path: Path) -> list[dict[str, Any]]:
        """Convert supported documents to text-only page records."""

        return get_loader(file_path).load(file_path)

    def _split_and_clean(
        self, pages: list[dict[str, Any]], source: str
    ) -> list[dict[str, Any]]:
        """Split page text into chunks, clean whitespace, and drop empty chunks."""

        chunks: list[dict[str, Any]] = []
        settings = getattr(self, "settings", global_settings)

        for page in pages:
            page_no = page["page_number"]

            for chunk_index, raw_text in enumerate(self.splitter.split_text(page["text"])):
                cleaned = re.sub(r"\s+", " ", raw_text).strip()
                if settings.ENABLE_PII_REDACTION:
                    cleaned = redact_pii(cleaned)
                if not cleaned:
                    continue

                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        **page.get("metadata", {}),
                        "source": source,
                        "page_number": page_no,
                        "chunk_index": chunk_index,
                    },
                })

        return chunks
