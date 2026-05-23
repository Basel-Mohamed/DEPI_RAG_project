from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.preprocessing.preprocessing_service import DocumentProcessor


def test_document_processor_rejects_invalid_chunk_overlap() -> None:
    settings = SimpleNamespace(CHUNK_SIZE=100, CHUNK_OVERLAP=100)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        DocumentProcessor(settings=settings)


def test_split_and_clean_returns_normalized_chunks() -> None:
    processor = DocumentProcessor.__new__(DocumentProcessor)
    processor.splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
    )

    chunks = processor._split_and_clean(
        [{"text": "  First   paragraph.\n\nSecond paragraph.  ", "page_number": 3}],
        source="sample.pdf",
    )

    assert chunks
    assert chunks[0]["text"] == "First paragraph. Second paragraph."
    assert chunks[0]["metadata"] == {
        "source": "sample.pdf",
        "page_number": 3,
        "chunk_index": 0,
    }
