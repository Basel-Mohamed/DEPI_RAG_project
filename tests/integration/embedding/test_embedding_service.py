import pytest
from app.services.embedding.embedding_service import EmbeddingService
from app.core.config import Settings


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Single instance shared across all tests — model loads once."""
    test_settings = Settings(
        APP_NAME="test",
        APP_VERSION="0.0.0",
        EMBEDDING_MODEL="intfloat/multilingual-e5-large",
    )
    return EmbeddingService(settings=test_settings)


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {"text": "Docling extracts text from PDF documents.", "metadata": {"source": "doc.pdf", "page_number": 1}},
        {"text": "FastEmbed runs on ONNX and requires no GPU.", "metadata": {"source": "doc.pdf", "page_number": 2}},
        {"text": "Qdrant stores dense vectors for similarity search.", "metadata": {"source": "doc.pdf", "page_number": 3}},
    ]


# ------------------------------------------------------------------
# embed_chunks
# ------------------------------------------------------------------

def test_embed_chunks_returns_correct_count(embedding_service, sample_chunks):
    result = embedding_service.embed_chunks(sample_chunks)
    assert len(result) == len(sample_chunks)


def test_embed_chunks_preserves_text_and_metadata(embedding_service, sample_chunks):
    result = embedding_service.embed_chunks(sample_chunks)
    for original, embedded in zip(sample_chunks, result):
        assert embedded["text"] == original["text"]
        assert embedded["metadata"] == original["metadata"]


def test_embed_chunks_embedding_is_list_of_floats(embedding_service, sample_chunks):
    result = embedding_service.embed_chunks(sample_chunks)
    for chunk in result:
        assert "embedding" in chunk
        assert isinstance(chunk["embedding"], list)
        assert all(isinstance(v, float) for v in chunk["embedding"])


def test_embed_chunks_consistent_dimension(embedding_service, sample_chunks):
    """All embeddings must share the same vector dimension."""
    result = embedding_service.embed_chunks(sample_chunks)
    dims = {len(chunk["embedding"]) for chunk in result}
    assert len(dims) == 1, f"Inconsistent embedding dimensions: {dims}"


def test_embed_chunks_empty_input(embedding_service):
    assert embedding_service.embed_chunks([]) == []


# ------------------------------------------------------------------
# embed_query
# ------------------------------------------------------------------

def test_embed_query_returns_list_of_floats(embedding_service):
    result = embedding_service.embed_query("What is Qdrant?")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_query_dimension_matches_chunks(embedding_service, sample_chunks):
    """Query vector must be the same dimension as chunk vectors."""
    chunks_result = embedding_service.embed_chunks(sample_chunks)
    query_result = embedding_service.embed_query("PDF text extraction")

    chunk_dim = len(chunks_result[0]["embedding"])
    query_dim = len(query_result)
    assert chunk_dim == query_dim, (
        f"Dimension mismatch: chunk={chunk_dim}, query={query_dim}"
    )


def test_embed_query_similar_text_scores_higher(embedding_service):
    """A query should score higher against a related chunk than an unrelated one."""
    import numpy as np

    query = embedding_service.embed_query("PDF document processing")
    related = embedding_service.embed_chunks(
        [{"text": "Docling converts PDF files into structured text.", "metadata": {}}]
    )[0]["embedding"]
    unrelated = embedding_service.embed_chunks(
        [{"text": "The stock market closed higher on Tuesday.", "metadata": {}}]
    )[0]["embedding"]

    q = np.array(query)
    score_related = float(np.dot(q, np.array(related)))
    score_unrelated = float(np.dot(q, np.array(unrelated)))

    assert score_related > score_unrelated, (
        f"Expected related chunk to score higher. "
        f"related={score_related:.4f}, unrelated={score_unrelated:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
