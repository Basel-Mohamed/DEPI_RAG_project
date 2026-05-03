import re
import io
import logging
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings as global_settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.utils.export import generate_multimodal_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Load a PDF with Docling, split text into chunks, and return cleaned chunks.

    Page images are stored separately (keyed by page number) and only
    referenced in chunk metadata — not duplicated per chunk.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or global_settings

        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.image_scale = settings.IMAGE_SCALE
        self.image_format = settings.IMAGE_FORMAT.upper()

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.image_scale <= 0:
            raise ValueError("image_scale must be greater than 0.")
        if self.image_format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("image_format must be one of: PNG, JPEG, WEBP.")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.image_scale
        pipeline_options.generate_page_images = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Instantiated once and reused across calls.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def process_document(
        self, file_path: str | Path
    ) -> tuple[list[dict[str, Any]], dict[int, bytes]]:
        """Run the full pipeline for a single PDF.

        Returns:
            chunks     — list of {"text": ..., "metadata": {"source", "page_number", "chunk_index"}}
            page_images — dict mapping page_number (1-based) → encoded image bytes
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info("Processing document: %s", file_path)

        pages, page_images = self._load_pages(file_path)
        chunks = self._split_and_clean(pages, source=str(file_path))

        logger.info("Produced %d chunks from %d pages.", len(chunks), len(pages))
        return chunks, page_images

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pages(
        self, file_path: Path
    ) -> tuple[list[dict[str, Any]], dict[int, bytes]]:
        """Convert PDF to per-page text. Images are returned separately."""
        conv_res = self.doc_converter.convert(file_path)

        pages: list[dict[str, Any]] = []
        page_images: dict[int, bytes] = {}

        for page_result in generate_multimodal_pages(conv_res):
            # Unpack: generate_multimodal_pages yields a 6-element tuple;
            # we only need the first (text) and last (page object).
            content_text, *_, page = page_result
            page_no = page.page_no + 1  # Docling is 0-based; store as 1-based.

            if page.image:
                with io.BytesIO() as buf:
                    page.image.save(buf, format=self.image_format)
                    page_images[page_no] = buf.getvalue()

            pages.append({"text": content_text or "", "page_number": page_no})

        return pages, page_images

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