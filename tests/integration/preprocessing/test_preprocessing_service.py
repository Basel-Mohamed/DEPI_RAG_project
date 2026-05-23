from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.preprocessing.preprocessing_service import DocumentProcessor


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
        APP_NAME="test",
        APP_VERSION="0.0.0",
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=200,
    )
    return DocumentProcessor(settings=test_settings)


def test_document_processing(
    document_processor: DocumentProcessor, sample_pdf_path: Path
):
    """Verify that DocumentProcessor produces text chunks."""

    chunks = document_processor.process_document(sample_pdf_path)

    assert isinstance(chunks, list), "Expected a list of chunks."
    assert len(chunks) > 0, "Expected at least one chunk."

    for chunk in chunks:
        assert "text" in chunk, "Chunk missing 'text'."
        assert chunk["text"].strip(), "Chunk text should not be empty."

        meta = chunk["metadata"]
        assert "source" in meta, "Metadata missing 'source'."
        assert "page_number" in meta, "Metadata missing 'page_number'."
        assert "chunk_index" in meta, "Metadata missing 'chunk_index'."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
