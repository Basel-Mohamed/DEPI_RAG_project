from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Load documents with Docling and convert them into cleaned text chunks."""

    def __init__(self,settings: Settings = settings) -> None:
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Run full preprocessing pipeline for a single document path."""
        documents = self.load_document(file_path)
        chunks = self.split_into_chunks(documents)
        return self.preprocess_chunks(chunks)
    
    
    def load_document(self, file_path: str | Path) -> list[Any]:
        loader = DoclingLoader(file_path=str(file_path))
        return loader.load()

    def split_into_chunks(self, documents: list[Any]) -> list[dict[str, Any]]:
        """
        Split loaded LangChain documents into chunks using recursive splitting.

        Returns a list of dictionaries:
        {
            "text": <chunk_text>,
            "metadata": <source_document_metadata_with_chunk_index>
        }
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks: list[dict[str, Any]] = []
        for doc_index, document in enumerate(documents):
            text = getattr(document, "page_content", "") or ""
            metadata = dict(getattr(document, "metadata", {}) or {})

            for chunk_index, chunk_text in enumerate(splitter.split_text(text)):
                chunk_metadata = {
                    **metadata,
                    "document_index": doc_index,
                    "chunk_index": chunk_index,
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks

    def preprocess_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Clean chunk text:
        - normalize internal whitespace
        - strip leading/trailing spaces
        - drop empty chunks
        """
        cleaned_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            raw_text = chunk.get("text", "")
            cleaned_text = self._normalize_whitespace(raw_text)
            if not cleaned_text:
                continue

            cleaned_chunks.append(
                {
                    "text": cleaned_text,
                    "metadata": dict(chunk.get("metadata", {})),
                }
            )

        return cleaned_chunks


    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


