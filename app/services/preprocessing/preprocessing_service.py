import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings as global_settings
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.legacy_doc.base import Ref
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Load a PDF with Docling, split page text into chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or global_settings

        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def process_document(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Run the text-only processing pipeline for a single PDF."""

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info("Processing document: %s", file_path)

        pages = self._load_pages(file_path)
        chunks = self._split_and_clean(pages, source=str(file_path))

        logger.info("Produced %d chunks from %d pages.", len(chunks), len(pages))
        return chunks

    def _load_pages(self, file_path: Path) -> list[dict[str, Any]]:
        """Convert PDF pages to text-only page records."""

        conv_res = self.doc_converter.convert(file_path)
        doc = conv_res.legacy_document
        pages: list[dict[str, Any]] = []
        text_by_page: dict[int, list[str]] = {}

        if doc.main_text is None:
            return pages

        for original_item in doc.main_text:
            item = doc._resolve_ref(original_item) if isinstance(original_item, Ref) else original_item
            if item is None or item.prov is None or len(item.prov) == 0:
                continue

            text = getattr(item, "text", None)
            if not text:
                continue

            page_no = int(item.prov[0].page)
            text_by_page.setdefault(page_no, []).append(str(text))

        for page_no in sorted(text_by_page):
            pages.append({
                "text": " ".join(text_by_page[page_no]),
                "page_number": page_no,
            })

        return pages

    def _split_and_clean(
        self, pages: list[dict[str, Any]], source: str
    ) -> list[dict[str, Any]]:
        """Split page text into chunks, clean whitespace, and drop empty chunks."""

        chunks: list[dict[str, Any]] = []

        for page in pages:
            page_no = page["page_number"]

            for chunk_index, raw_text in enumerate(self.splitter.split_text(page["text"])):
                cleaned = re.sub(r"\s+", " ", raw_text).strip()
                if not cleaned:
                    continue

                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        "source": source,
                        "page_number": page_no,
                        "chunk_index": chunk_index,
                    },
                })

        return chunks
