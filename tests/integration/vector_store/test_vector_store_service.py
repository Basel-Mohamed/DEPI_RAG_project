"""
Integration tests for QdrantService.

Uses QdrantClient(":memory:") — no Docker or running Qdrant instance required.
Each test gets a fresh in-memory client via the function-scoped fixture.

Run with:
    pytest tests/integration/vector_store/test_qdrant_store.py -v
"""
import pytest
from app.core.config import Settings
from app.services.embedding.embedding_service import EmbeddingService
from app.services.vectorstore.qdrant_store import QdrantService, SearchMode


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Load the embedding model once for the entire module."""
    test_settings = Settings(
        APP_NAME="test",
        APP_VERSION="0.0.0",
        EMBEDDING_MODEL="intfloat/multilingual-e5-large",
        SPARSE_MODEL="Qdrant/bm25",
    )
    return EmbeddingService(settings=test_settings)


@pytest.fixture(scope="module")
def raw_chunks() -> list[dict]:
    """Sample chunks that reflect real DocumentProcessor output."""
    return [
        {
            "text": "Digital marketing involves promoting products through online channels.",
            "metadata": {"source": "marketing.pdf", "page_number": 1, "chunk_index": 0},
        },
        {
            "text": "Social media platforms are essential tools for brand awareness.",
            "metadata": {"source": "marketing.pdf", "page_number": 1, "chunk_index": 1},
        },
        {
            "text": "Search engine optimization improves organic website traffic.",
            "metadata": {"source": "seo.pdf", "page_number": 1, "chunk_index": 0},
        },
        {
            "text": "Keyword research is the foundation of any SEO strategy.",
            "metadata": {"source": "seo.pdf", "page_number": 2, "chunk_index": 0},
        },
        {
            "text": "Content marketing builds trust and drives long-term engagement.",
            "metadata": {"source": "marketing.pdf", "page_number": 2, "chunk_index": 0},
        },
    ]


@pytest.fixture(scope="module")
def embedded_chunks(embedding_service, raw_chunks) -> list[dict]:
    """Embed sample chunks once — reused across all tests."""
    return embedding_service.embed_chunks(raw_chunks)


@pytest.fixture
def qdrant_service(embedding_service) -> QdrantService:
    """
    Fresh in-memory QdrantService per test.
    ':memory:' gives each test a fully isolated Qdrant instance.
    """
    test_settings = Settings(
        APP_NAME="test",
        APP_VERSION="0.0.0",
        EMBEDDING_MODEL="intfloat/multilingual-e5-large",
        SPARSE_MODEL="Qdrant/bm25",
        QDRANT_REMOTE=False,
        QDRANT_PATH=":memory:",
        COLLECTION_NAME="test_collection",
        DENSE_VECTOR_SIZE=1024,
        RETRIEVAL_MODE="hybrid",
        SCORE_THRESHOLD=0.0,
    )
    return QdrantService(embedding_service=embedding_service, settings=test_settings)


@pytest.fixture
def populated_qdrant(qdrant_service, embedded_chunks) -> QdrantService:
    """QdrantService with sample data already upserted."""
    qdrant_service.upsert(embedded_chunks)
    return qdrant_service


# ------------------------------------------------------------------
# Collection setup
# ------------------------------------------------------------------

def test_collection_is_created_on_init(qdrant_service):
    assert qdrant_service.client.collection_exists(qdrant_service.collection)


def test_ensure_collection_is_idempotent(qdrant_service):
    """Calling _ensure_collection twice must not raise or duplicate."""
    qdrant_service._ensure_collection()
    info = qdrant_service.client.get_collection(qdrant_service.collection)
    assert info is not None


def test_collection_has_dense_and_sparse_vectors(qdrant_service):
    info = qdrant_service.client.get_collection(qdrant_service.collection)
    named_vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors

    assert QdrantService.DENSE_VECTOR_NAME in named_vectors
    assert QdrantService.SPARSE_VECTOR_NAME in sparse_vectors


# ------------------------------------------------------------------
# Health check & collection info
# ------------------------------------------------------------------

def test_health_check_returns_healthy(qdrant_service):
    result = qdrant_service.health_check()
    assert result["healthy"] is True
    assert result["collection"] == "test_collection"
    assert "points_count" in result


def test_collection_info_structure(populated_qdrant, embedded_chunks):
    info = populated_qdrant.collection_info()
    assert info["collection"] == "test_collection"
    assert info["points_count"] == len(embedded_chunks)
    assert "status" in info
    assert info["config"]["dense_size"] == 1024


# ------------------------------------------------------------------
# Upsert
# ------------------------------------------------------------------

def test_upsert_returns_correct_count(qdrant_service, embedded_chunks):
    result = qdrant_service.upsert(embedded_chunks)
    assert result["upserted"] == len(embedded_chunks)
    assert result["failed"] == 0


def test_upsert_empty_list_returns_zero(qdrant_service):
    result = qdrant_service.upsert([])
    assert result == {"upserted": 0, "failed": 0}


def test_upsert_is_idempotent(qdrant_service, embedded_chunks):
    """Upserting the same chunks twice must not create duplicates."""
    qdrant_service.upsert(embedded_chunks)
    qdrant_service.upsert(embedded_chunks)
    count = qdrant_service.client.count(qdrant_service.collection).count
    assert count == len(embedded_chunks)


def test_upsert_points_are_retrievable(populated_qdrant, embedded_chunks):
    count = populated_qdrant.client.count(populated_qdrant.collection).count
    assert count == len(embedded_chunks)


# ------------------------------------------------------------------
# get_by_ids
# ------------------------------------------------------------------

def test_get_by_ids_returns_correct_points(populated_qdrant, embedded_chunks):
    # Build the same deterministic IDs that _build_points would produce
    meta = embedded_chunks[0]["metadata"]
    raw_key = f"{meta['source']}::{meta['page_number']}::{meta['chunk_index']}"
    point_id = QdrantService._to_uuid(raw_key)

    results = populated_qdrant.get_by_ids([raw_key])
    assert len(results) == 1
    assert results[0]["id"] == point_id
    assert results[0]["text"] == embedded_chunks[0]["text"]


def test_get_by_ids_missing_id_returns_empty(populated_qdrant):
    results = populated_qdrant.get_by_ids(["nonexistent::99::99"])
    assert results == []


# ------------------------------------------------------------------
# update_payload
# ------------------------------------------------------------------

def test_update_payload_merges_new_field(populated_qdrant, embedded_chunks):
    meta = embedded_chunks[0]["metadata"]
    raw_key = f"{meta['source']}::{meta['page_number']}::{meta['chunk_index']}"

    populated_qdrant.update_payload(raw_key, {"reviewed": True})

    results = populated_qdrant.get_by_ids([raw_key])
    assert results[0]["metadata"].get("reviewed") is True


def test_update_payload_preserves_existing_fields(populated_qdrant, embedded_chunks):
    meta = embedded_chunks[0]["metadata"]
    raw_key = f"{meta['source']}::{meta['page_number']}::{meta['chunk_index']}"

    populated_qdrant.update_payload(raw_key, {"tag": "important"})

    results = populated_qdrant.get_by_ids([raw_key])
    assert results[0]["metadata"]["source"] == meta["source"]
    assert results[0]["metadata"]["tag"] == "important"


# ------------------------------------------------------------------
# delete_by_filter
# ------------------------------------------------------------------

def test_delete_by_filter_removes_matching_points(populated_qdrant, embedded_chunks):
    marketing_count = sum(
        1 for c in embedded_chunks if c["metadata"]["source"] == "marketing.pdf"
    )
    result = populated_qdrant.delete_by_filter("source", "marketing.pdf")

    assert result["deleted_count"] == marketing_count
    remaining = populated_qdrant.client.count(populated_qdrant.collection).count
    assert remaining == len(embedded_chunks) - marketing_count


def test_delete_by_filter_list_value(populated_qdrant, embedded_chunks):
    """Filter with a list of values should delete all matching."""
    result = populated_qdrant.delete_by_filter(
        "source", ["marketing.pdf", "seo.pdf"]
    )
    assert result["deleted_count"] == len(embedded_chunks)
    assert populated_qdrant.client.count(populated_qdrant.collection).count == 0


def test_delete_by_filter_no_match_deletes_nothing(populated_qdrant, embedded_chunks):
    result = populated_qdrant.delete_by_filter("source", "does_not_exist.pdf")
    assert result["deleted_count"] == 0
    assert populated_qdrant.client.count(populated_qdrant.collection).count == len(embedded_chunks)


# ------------------------------------------------------------------
# delete_by_ids
# ------------------------------------------------------------------

def test_delete_by_ids_removes_specific_points(populated_qdrant, embedded_chunks):
    meta = embedded_chunks[0]["metadata"]
    raw_key = f"{meta['source']}::{meta['page_number']}::{meta['chunk_index']}"

    result = populated_qdrant.delete_by_ids([raw_key])
    assert result["deleted_count"] == 1

    remaining = populated_qdrant.get_by_ids([raw_key])
    assert remaining == []


# ------------------------------------------------------------------
# scroll
# ------------------------------------------------------------------

def test_scroll_no_filter_returns_all(populated_qdrant, embedded_chunks):
    results = populated_qdrant.scroll(limit=100)
    assert len(results) == len(embedded_chunks)


def test_scroll_with_filter_returns_only_matching(populated_qdrant, embedded_chunks):
    seo_count = sum(
        1 for c in embedded_chunks if c["metadata"]["source"] == "seo.pdf"
    )
    results = populated_qdrant.scroll(filter_field="source", filter_value="seo.pdf")
    assert len(results) == seo_count
    assert all(r["metadata"]["source"] == "seo.pdf" for r in results)


def test_scroll_result_structure(populated_qdrant):
    results = populated_qdrant.scroll(limit=1)
    assert len(results) == 1
    point = results[0]
    assert "id" in point
    assert "text" in point
    assert "metadata" in point
    assert "source" in point["metadata"]
    assert "page_number" in point["metadata"]


# ------------------------------------------------------------------
# Search — dense
# ------------------------------------------------------------------

def test_search_dense_returns_results(populated_qdrant):
    results = populated_qdrant.search("marketing strategy", mode=SearchMode.DENSE)
    assert len(results) > 0


def test_search_dense_result_structure(populated_qdrant):
    results = populated_qdrant.search("marketing", mode=SearchMode.DENSE, top_k=1)
    assert len(results) == 1
    result = results[0]
    assert "id" in result
    assert "text" in result
    assert "score" in result
    assert "metadata" in result
    assert isinstance(result["score"], float)


def test_search_dense_top_k_respected(populated_qdrant, embedded_chunks):
    top_k = 2
    results = populated_qdrant.search("marketing", mode=SearchMode.DENSE, top_k=top_k)
    assert len(results) <= top_k


# ------------------------------------------------------------------
# Search — sparse (BM25)
# ------------------------------------------------------------------

def test_search_sparse_returns_results(populated_qdrant):
    results = populated_qdrant.search("SEO keyword", mode=SearchMode.SPARSE)
    assert len(results) > 0


def test_search_sparse_result_structure(populated_qdrant):
    results = populated_qdrant.search("search engine", mode=SearchMode.SPARSE, top_k=1)
    assert len(results) == 1
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


def test_search_sparse_top_k_respected(populated_qdrant):
    results = populated_qdrant.search("marketing", mode=SearchMode.SPARSE, top_k=2)
    assert len(results) <= 2


# ------------------------------------------------------------------
# Search — hybrid (RRF)
# ------------------------------------------------------------------

def test_search_hybrid_returns_results(populated_qdrant):
    results = populated_qdrant.search("digital marketing strategy", mode=SearchMode.HYBRID)
    assert len(results) > 0


def test_search_hybrid_top_k_respected(populated_qdrant):
    results = populated_qdrant.search("marketing", mode=SearchMode.HYBRID, top_k=3)
    assert len(results) <= 3


def test_search_hybrid_result_structure(populated_qdrant):
    results = populated_qdrant.search("SEO", mode=SearchMode.HYBRID, top_k=1)
    assert len(results) == 1
    assert all(k in results[0] for k in ("id", "text", "score", "metadata"))


# ------------------------------------------------------------------
# Search — with filter
# ------------------------------------------------------------------

def test_search_with_filter_limits_to_source(populated_qdrant):
    results = populated_qdrant.search(
        "content strategy",
        mode=SearchMode.HYBRID,
        filter_field="source",
        filter_value="seo.pdf",
    )
    assert all(r["metadata"]["source"] == "seo.pdf" for r in results)


def test_search_with_list_filter(populated_qdrant):
    results = populated_qdrant.search(
        "marketing",
        mode=SearchMode.DENSE,
        filter_field="source",
        filter_value=["marketing.pdf"],
    )
    assert all(r["metadata"]["source"] == "marketing.pdf" for r in results)


def test_search_default_mode_from_settings(populated_qdrant):
    """search() with no mode arg should use settings.RETRIEVAL_MODE (hybrid)."""
    results = populated_qdrant.search("digital marketing")
    assert len(results) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


# ------------------------------------------------------------------
# Metrics instrumentation
# ------------------------------------------------------------------

def test_upsert_records_qdrant_latency(qdrant_service, embedded_chunks):
    """QDRANT_LATENCY histogram should have observations after upsert."""
    from app.core.metrics import QDRANT_LATENCY

    before = QDRANT_LATENCY._sum.get()
    qdrant_service.upsert(embedded_chunks)
    after = QDRANT_LATENCY._sum.get()

    assert after > before


def test_upsert_increments_chunks_ingested(qdrant_service, embedded_chunks):
    """CHUNKS_INGESTED counter should increment by the number of chunks upserted."""
    from app.core.metrics import CHUNKS_INGESTED

    before = CHUNKS_INGESTED._value.get()
    qdrant_service.upsert(embedded_chunks)
    after = CHUNKS_INGESTED._value.get()

    assert after == before + len(embedded_chunks)


def test_upsert_empty_does_not_increment_chunks(qdrant_service):
    """CHUNKS_INGESTED should not change when upserting empty list."""
    from app.core.metrics import CHUNKS_INGESTED

    before = CHUNKS_INGESTED._value.get()
    qdrant_service.upsert([])
    after = CHUNKS_INGESTED._value.get()

    assert after == before


def test_dense_search_records_qdrant_latency(populated_qdrant):
    """QDRANT_LATENCY should increase after a dense search."""
    from app.core.metrics import QDRANT_LATENCY

    before = QDRANT_LATENCY._sum.get()
    populated_qdrant.search("marketing", mode=SearchMode.DENSE, top_k=2)
    after = QDRANT_LATENCY._sum.get()

    assert after > before


def test_sparse_search_records_qdrant_latency(populated_qdrant):
    """QDRANT_LATENCY should increase after a sparse search."""
    from app.core.metrics import QDRANT_LATENCY

    before = QDRANT_LATENCY._sum.get()
    populated_qdrant.search("SEO keyword", mode=SearchMode.SPARSE, top_k=2)
    after = QDRANT_LATENCY._sum.get()

    assert after > before


def test_hybrid_search_records_qdrant_latency(populated_qdrant):
    """QDRANT_LATENCY should increase after a hybrid search."""
    from app.core.metrics import QDRANT_LATENCY

    before = QDRANT_LATENCY._sum.get()
    populated_qdrant.search("digital marketing", mode=SearchMode.HYBRID, top_k=2)
    after = QDRANT_LATENCY._sum.get()

    assert after > before


def test_health_check_sets_qdrant_up_gauge(qdrant_service):
    """health_check() should set QDRANT_UP to 1 when healthy."""
    from app.core.metrics import QDRANT_UP

    qdrant_service.health_check()
    assert QDRANT_UP._value.get() == 1.0


def test_health_check_updates_qdrant_up_on_failure(qdrant_service):
    """QDRANT_UP should be set to 0 when health check fails."""
    from app.core.metrics import QDRANT_UP
    from unittest.mock import patch

    with patch.object(
        qdrant_service.client,
        "get_collection",
        side_effect=Exception("connection failed"),
    ):
        qdrant_service.health_check()

    assert QDRANT_UP._value.get() == 0.0