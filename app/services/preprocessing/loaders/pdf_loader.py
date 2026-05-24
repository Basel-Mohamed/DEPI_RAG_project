from pathlib import Path
from typing import Any

from app.services.preprocessing.loaders.base_loader import BaseLoader
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.legacy_doc.base import Ref


class PdfLoader(BaseLoader):
    def __init__(self) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        self.doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def load(self, file_path: Path) -> list[dict[str, Any]]:
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
                "metadata": {},
            })

        return pages
