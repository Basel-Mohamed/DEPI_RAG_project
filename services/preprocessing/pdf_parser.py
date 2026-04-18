import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from domain.models.document import Document


class PDFParser:
    def __init__(self, asset_root: str = "data/ingestion_assets"):
        self.asset_root = Path(asset_root)

    def parse(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        document_id = self._build_document_id(path)

        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path, document_id)

        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return []

        return [
            Document(
                page_content=text,
                metadata=self._build_metadata(
                    path=path,
                    document_id=document_id,
                    modality="text",
                ),
            )
        ]

    def _parse_pdf(self, path: Path, document_id: str) -> list[Document]:
        try:
            return self._parse_pdf_with_fitz(path, document_id)
        except ImportError:
            return self._parse_pdf_with_pypdf(path, document_id)

    def _parse_pdf_with_fitz(self, path: Path, document_id: str) -> list[Document]:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("PyMuPDF is required for image-aware PDF parsing.") from exc

        documents: list[Document] = []
        asset_dir = self.asset_root / document_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        pdf = fitz.open(path)
        total_pages = pdf.page_count

        for page_index in range(total_pages):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            page_text = (page.get_text("text") or "").strip()

            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata=self._build_metadata(
                            path=path,
                            document_id=document_id,
                            modality="text",
                            page_number=page_number,
                            total_pages=total_pages,
                        ),
                    )
                )

            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                extracted_image = pdf.extract_image(xref)
                image_bytes = extracted_image.get("image", b"")

                if not image_bytes:
                    continue

                image_extension = extracted_image.get("ext", "bin")
                image_name = (
                    f"page-{page_number:04d}-image-{image_index:03d}.{image_extension}"
                )
                image_path = self.asset_root / document_id / image_name
                image_path.write_bytes(image_bytes)

                documents.append(
                    Document(
                        page_content=self._build_image_summary(
                            file_name=path.name,
                            page_number=page_number,
                            image_index=image_index,
                            page_text=page_text,
                        ),
                        metadata=self._build_metadata(
                            path=path,
                            document_id=document_id,
                            modality="image",
                            page_number=page_number,
                            total_pages=total_pages,
                            image_index=image_index,
                            image_path=str(image_path),
                            image_mime_type=(
                                mimetypes.guess_type(image_path.name)[0]
                                or "application/octet-stream"
                            ),
                        ),
                    )
                )

        pdf.close()
        return documents

    def _parse_pdf_with_pypdf(self, path: Path, document_id: str) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "Install 'PyMuPDF' for text and image extraction or 'pypdf' for text-only PDFs."
            ) from exc

        documents: list[Document] = []
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata=self._build_metadata(
                        path=path,
                        document_id=document_id,
                        modality="text",
                        page_number=page_number,
                        total_pages=total_pages,
                    ),
                )
            )

        return documents

    @staticmethod
    def _build_document_id(path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest[:24]

    @staticmethod
    def _build_image_summary(
        file_name: str,
        page_number: int,
        image_index: int,
        page_text: str,
    ) -> str:
        normalized_page_text = " ".join(page_text.split())
        if normalized_page_text:
            context = normalized_page_text[:1000]
            return (
                f"Image {image_index} extracted from page {page_number} of {file_name}. "
                f"Nearby page text: {context}"
            )

        return (
            f"Image {image_index} extracted from page {page_number} of {file_name}. "
            "No nearby text was available on the page."
        )

    @staticmethod
    def _build_metadata(
        path: Path,
        document_id: str,
        modality: str,
        page_number: int | None = None,
        total_pages: int | None = None,
        image_index: int | None = None,
        image_path: str | None = None,
        image_mime_type: str | None = None,
    ) -> dict:
        metadata = {
            "document_id": document_id,
            "source": str(path),
            "file_name": path.name,
            "file_type": path.suffix.lower().lstrip(".") or "unknown",
            "modality": modality,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "parser": "pdf_parser",
        }

        if page_number is not None:
            metadata["page_number"] = page_number

        if total_pages is not None:
            metadata["total_pages"] = total_pages

        if image_index is not None:
            metadata["image_index"] = image_index

        if image_path is not None:
            metadata["image_path"] = image_path

        if image_mime_type is not None:
            metadata["image_mime_type"] = image_mime_type

        return metadata
