import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings as global_settings
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Load a PDF with Docling, split page text into chunks, and return cleaned chunks."""

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
        """Run the full text-only pipeline for a single PDF."""

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info("Processing document: %s", file_path)

        pages = self._load_pages(file_path)
        chunks = self._split_and_clean(pages, source=str(file_path))

        logger.info("Produced %d chunks from %d pages.", len(chunks), len(pages))
        return chunks

    def _load_pages(self, file_path: Path) -> list[dict[str, Any]]:
        """Convert PDF to per-page text."""

        conv_res = self.doc_converter.convert(file_path)
        pages: list[dict[str, Any]] = []

        for page_result in generate_multimodal_pages(conv_res):
            content_text, *_, page = page_result
            page_no = page.page_no + 1
            pages.append({"text": content_text or "", "page_number": page_no})

        return pages

    def _split_and_clean(
        self,
        pages: list[dict[str, Any]],
        source: str,
    ) -> list[dict[str, Any]]:
        """Split page text into chunks, clean whitespace, and drop empty chunks."""

        chunks: list[dict[str, Any]] = []

        for page in pages:
            page_no = page["page_number"]
            for chunk_index, raw_text in enumerate(self.splitter.split_text(page["text"])):
                cleaned = re.sub(r"\s+", " ", raw_text).strip()
                if not cleaned:
                    continue

                chunks.append(
                    {
                        "text": cleaned,
                        "metadata": {
                            "source": source,
                            "page_number": page_no,
                            "chunk_index": chunk_index,
                        },
                    }
                )

        return chunks
