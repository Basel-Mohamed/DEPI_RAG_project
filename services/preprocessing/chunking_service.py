from domain.models.document import Document


class _FallbackTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunks.append(text[start:end])

            if end == text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks


class ChunkingService:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            RecursiveCharacterTextSplitter = None

        if RecursiveCharacterTextSplitter is None:
            self.splitter = _FallbackTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    @staticmethod
    def _build_chunk_id(metadata: dict, chunk_index: int) -> str:
        page_number = metadata.get("page_number", 0)
        modality = metadata.get("modality", "text")
        image_index = metadata.get("image_index", 0)

        return (
            f"{metadata['document_id']}:"
            f"page:{page_number}:"
            f"modality:{modality}:"
            f"image:{image_index}:"
            f"chunk:{chunk_index}"
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunked_documents: list[Document] = []

        for document in documents:
            text_chunks = self.splitter.split_text(document.page_content)

            if not text_chunks and document.page_content.strip():
                text_chunks = [document.page_content]

            for chunk_index, chunk in enumerate(text_chunks):
                cleaned_chunk = chunk.strip()
                if not cleaned_chunk:
                    continue

                chunk_metadata = dict(document.metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["chunk_size"] = len(cleaned_chunk)
                chunk_metadata["chunk_id"] = self._build_chunk_id(
                    chunk_metadata,
                    chunk_index,
                )

                chunked_documents.append(
                    Document(page_content=cleaned_chunk, metadata=chunk_metadata)
                )

        return chunked_documents
