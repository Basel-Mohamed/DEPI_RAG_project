import io
import pytest
from pathlib import Path
from PIL import Image

from app.services.preprocessing.preprocessing_service import DocumentProcessor
from app.core.config import Settings


@pytest.fixture
def sample_pdf_path() -> Path:
    test_dir = Path(__file__).parent
    pdf_path = test_dir / "data" / "sample_test.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture
def document_processor() -> DocumentProcessor:
    test_settings = Settings(
        APP_NAME="test",           # required — no default in config
        APP_VERSION="0.0.0",       # required — no default in config
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=200,
        IMAGE_SCALE=2.0,
        IMAGE_FORMAT="PNG",
    )
    return DocumentProcessor(settings=test_settings)


def test_document_processing(
    document_processor: DocumentProcessor, sample_pdf_path: Path
):
    """
    Integration test: verify that DocumentProcessor produces valid text chunks
    and extracts uncorrupted page images from a real PDF.
    """
    # 1. Execute the pipeline — now returns (chunks, page_images)
    chunks, page_images = document_processor.process_document(sample_pdf_path)

    # 2. Basic chunk assertions
    assert isinstance(chunks, list), "Expected a list of chunks."
    assert len(chunks) > 0, "Expected at least one chunk."

    # 3. Per-chunk structure
    for chunk in chunks:
        assert "text" in chunk, "Chunk missing 'text'."
        assert chunk["text"].strip(), "Chunk text should not be empty."

        meta = chunk["metadata"]
        assert "source" in meta, "Metadata missing 'source'."
        assert "page_number" in meta, "Metadata missing 'page_number'."
        assert "chunk_index" in meta, "Metadata missing 'chunk_index'."

    # 4. Page images are stored separately, keyed by page number
    assert isinstance(page_images, dict), "Expected page_images to be a dict."
    assert len(page_images) > 0, "Expected at least one page image."

    # 5. Verify every image is valid and uncorrupted
    for page_no, img_bytes in page_images.items():
        assert isinstance(img_bytes, bytes) and len(img_bytes) > 0, (
            f"Page {page_no}: image bytes are empty."
        )
        try:
            image = Image.open(io.BytesIO(img_bytes))
            image.verify()
        except Exception as e:
            pytest.fail(f"Page {page_no}: image bytes are corrupted — {e}")

    # 6. Every chunk's page_number must have a corresponding image
    chunk_page_numbers = {chunk["metadata"]["page_number"] for chunk in chunks}
    missing = chunk_page_numbers - page_images.keys()
    assert not missing, f"Chunks reference pages with no extracted image: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])