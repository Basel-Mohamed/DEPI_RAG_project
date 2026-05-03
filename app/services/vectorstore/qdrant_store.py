import uuid
import logging
from enum import Enum
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    Fusion,
    FusionQuery,
    Prefetch,
)

from app.core.config import Settings, settings as global_settings
from app.services.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    DENSE = "dense"      # vector similarity only
    SPARSE = "sparse"    # BM25 keyword only
    HYBRID = "hybrid"    # RRF fusion of both


class QdrantService:
    """
    Qdrant vector store with full CRUD and three search modes.

    Upsert flow:
        chunks (embedded) → upsert points → verify → rollback on failure

    Search modes (controlled via settings or per-call override):
        dense  — cosine similarity on dense vectors
        sparse — BM25 keyword search on sparse vectors
        hybrid — RRF fusion of both (best quality)
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or global_settings
        self.collection = self.settings.COLLECTION_NAME
        self.embedding_service = embedding_service

        # Sparse model for BM25
        self.sparse_model = SparseTextEmbedding(
            model_name=self.settings.SPARSE_MODEL  # e.g. "Qdrant/bm25"
        )

        # Connect to Qdrant (remote or local)
        self.client = self._connect()

        # Create collection if it doesn't exist
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self) -> QdrantClient:
        try:
            if self.settings.QDRANT_REMOTE:
                client = QdrantClient(
                    host=self.settings.QDRANT_HOST,
                    port=self.settings.QDRANT_PORT,
                    prefer_grpc=self.settings.QDRANT_PREFER_GRPC,
                )
                logger.info("Connected to remote Qdrant at %s:%s", self.settings.QDRANT_HOST, self.settings.QDRANT_PORT)
            else:
                client = QdrantClient(path=self.settings.QDRANT_PATH)
                logger.info("Connected to local Qdrant at %s", self.settings.QDRANT_PATH)
            return client
        except Exception as e:
            logger.error("Failed to connect to Qdrant: %s", e)
            raise

    def _ensure_collection(self) -> None:
        """Create the collection with HNSW + quantization if it doesn't exist."""
        if self.client.collection_exists(self.collection):
            logger.info("Collection '%s' already exists.", self.collection)
            return

        hnsw = HnswConfigDiff(
            m=self.settings.HNSW_M,
            ef_construct=self.settings.HNSW_EF_CONSTRUCT,
            full_scan_threshold=self.settings.HNSW_FULL_SCAN_THRESHOLD,
            on_disk=self.settings.HNSW_ON_DISK,
        )
        optimizers = OptimizersConfigDiff(
            indexing_threshold=self.settings.OPTIMIZATION_INDEXING_THRESHOLD,
            memmap_threshold=self.settings.OPTIMIZATION_MEMMAP_THRESHOLD,
        )
        quantization = ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=self.settings.QUANTIZATION_QUANTILE,
                always_ram=self.settings.QUANTIZATION_ALWAYS_RAM,
            )
        )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.settings.DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=self.settings.DENSE_ON_DISK,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=self.settings.SPARSE_ON_DISK)
                )
            },
            hnsw_config=hnsw,
            optimizers_config=optimizers,
            quantization_config=quantization,
        )
        logger.info("Created collection '%s'.", self.collection)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, embedded_chunks: list[dict[str, Any]]) -> dict:
        """
        Upsert embedded chunks into Qdrant.
        Rolls back any partially inserted document on failure.

        Args:
            embedded_chunks: output of EmbeddingService.embed_chunks()
                Each chunk must have: "text", "embedding", "metadata"
                metadata must have: "source", "page_number", "chunk_index"

        Returns:
            {"upserted": int, "failed": int}
        """
        if not embedded_chunks:
            logger.warning("upsert called with empty chunk list.")
            return {"upserted": 0, "failed": 0}

        points = self._build_points(embedded_chunks)
        point_ids = [p.id for p in points]

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
        except Exception as e:
            logger.error("Upsert failed, attempting rollback: %s", e)
            self._rollback(point_ids)
            raise

        # Verify
        fetched = self.client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=False,
            with_vectors=False,
        )
        upserted_ids = {str(p.id) for p in fetched}
        failed_ids = [pid for pid in point_ids if str(pid) not in upserted_ids]

        if failed_ids:
            logger.warning("Rolling back %d partially upserted points.", len(failed_ids))
            self._rollback(point_ids)
            return {"upserted": 0, "failed": len(point_ids)}

        logger.info("Upserted %d chunks into '%s'.", len(points), self.collection)
        return {"upserted": len(points), "failed": 0}

    def get_by_ids(self, point_ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve specific points by their IDs."""
        results = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._to_uuid(pid) for pid in point_ids],
            with_payload=True,
            with_vectors=False,
        )
        return [self._format_point(p) for p in results]

    def update_payload(self, point_id: str, payload: dict[str, Any]) -> None:
        """Update (merge) the payload of a single point."""
        self.client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=[self._to_uuid(point_id)],
            wait=True,
        )
        logger.info("Updated payload for point %s.", point_id)

    def delete_by_filter(self, filter_field: str, filter_value: Any) -> dict:
        """
        Delete all points matching a metadata field/value pair.

        Args:
            filter_field: metadata key, e.g. "source"
            filter_value: scalar or list of values

        Returns:
            {"deleted_count": int}
        """
        count_before = self.client.count(self.collection).count

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=self._build_filter(filter_field, filter_value)
            ),
            wait=True,
        )

        count_after = self.client.count(self.collection).count
        deleted = count_before - count_after
        logger.info("Deleted %d points from '%s'.", deleted, self.collection)
        return {"deleted_count": deleted}

    def delete_by_ids(self, point_ids: list[str]) -> dict:
        """Delete specific points by their IDs."""
        uuids = [self._to_uuid(pid) for pid in point_ids]
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=uuids),
            wait=True,
        )
        logger.info("Deleted %d points by ID.", len(uuids))
        return {"deleted_count": len(uuids)}

    def scroll(
        self,
        filter_field: str | None = None,
        filter_value: Any = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Retrieve points by metadata filter without a query vector.
        Useful for metadata-only lookups (e.g. all chunks for a document).
        """
        scroll_filter = (
            self._build_filter(filter_field, filter_value)
            if filter_field
            else None
        )
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._format_point(p) for p in points]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: SearchMode | None = None,
        filter_field: str | None = None,
        filter_value: Any = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the vector store.

        Args:
            query:            The query string.
            top_k:            Number of results to return.
            mode:             SearchMode.DENSE | SPARSE | HYBRID.
                              Defaults to settings.RETRIEVAL_MODE.
            filter_field:     Optional metadata key to filter on.
            filter_value:     Scalar or list value for the filter.
            score_threshold:  Optional minimum score cutoff.

        Returns:
            List of {"id", "text", "score", "metadata"} dicts.
        """
        mode = mode or SearchMode(self.settings.RETRIEVAL_MODE)
        query_filter = (
            self._build_filter(filter_field, filter_value)
            if filter_field
            else None
        )
        score_threshold = score_threshold if score_threshold is not None else self.settings.SCORE_THRESHOLD

        if mode == SearchMode.DENSE:
            results = self._search_dense(query, top_k, query_filter, score_threshold)
        elif mode == SearchMode.SPARSE:
            results = self._search_sparse(query, top_k, query_filter, score_threshold)
        else:
            results = self._search_hybrid(query, top_k, query_filter, score_threshold)

        logger.info(
            "Search [%s] for '%s...' → %d results.", mode, query[:40], len(results)
        )
        return results

    def _search_dense(
        self,
        query: str,
        top_k: int,
        query_filter: Filter | None,
        score_threshold: float | None,
    ) -> list[dict[str, Any]]:
        dense_vec = self.embedding_service.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection,
            query=dense_vec,
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [self._format_scored(r) for r in results.points]

    def _search_sparse(
        self,
        query: str,
        top_k: int,
        query_filter: Filter | None,
        score_threshold: float | None,
    ) -> list[dict[str, Any]]:
        sparse_vec = next(self.sparse_model.query_embed(query))
        results = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist(),
            ),
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [self._format_scored(r) for r in results.points]

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        query_filter: Filter | None,
        score_threshold: float | None,
    ) -> list[dict[str, Any]]:
        """RRF fusion of dense + sparse results."""
        dense_vec = self.embedding_service.embed_query(query)
        sparse_vec = next(self.sparse_model.query_embed(query))

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=self.DENSE_VECTOR_NAME,
                    limit=top_k * 2,  # overfetch so RRF has enough to rank
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [self._format_scored(r) for r in results.points]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Return collection status and point count."""
        try:
            info = self.client.get_collection(self.collection)
            return {
                "status": str(info.status),
                "points_count": info.points_count,
                "collection": self.collection,
                "healthy": info.status in [
                    models.CollectionStatus.GREEN,
                    models.CollectionStatus.YELLOW,
                ],
            }
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {"healthy": False, "error": str(e)}

    def collection_info(self) -> dict:
        """Return detailed collection stats."""
        info = self.client.get_collection(self.collection)
        vectors_count = getattr(info, "vectors_count", None)
        return {
            "collection": self.collection,
            "points_count": info.points_count,
            "vectors_count": vectors_count,
            "status": str(info.status),
            "config": {
                "dense_size": self.settings.DENSE_VECTOR_SIZE,
                "retrieval_mode": self.settings.RETRIEVAL_MODE,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_points(self, embedded_chunks: list[dict[str, Any]]) -> list[PointStruct]:
        """Convert embedded chunks to Qdrant PointStructs with dense + sparse vectors."""
        texts = [c["text"] for c in embedded_chunks]
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for chunk, sparse_vec in zip(embedded_chunks, sparse_embeddings):
            meta = chunk["metadata"]
            # Deterministic ID from source + page + chunk index
            point_id = self._to_uuid(
                f"{meta['source']}::{meta['page_number']}::{meta['chunk_index']}"
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_VECTOR_NAME: chunk["embedding"],
                        self.SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "text": chunk["text"],
                        "source": meta["source"],
                        "page_number": meta["page_number"],
                        "chunk_index": meta["chunk_index"],
                    },
                )
            )
        return points

    def _build_filter(self, field: str, value: Any) -> Filter:
        """Build a Qdrant Filter for a single field/value pair."""
        condition = (
            FieldCondition(key=field, match=MatchAny(any=value))
            if isinstance(value, list)
            else FieldCondition(key=field, match=MatchValue(value=value))
        )
        return Filter(must=[condition])

    def _rollback(self, point_ids: list) -> None:
        """Delete a list of point IDs — used to clean up on upsert failure."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )
            logger.warning("Rollback successful: deleted %d points.", len(point_ids))
        except Exception as e:
            logger.error("Rollback failed: %s — manual cleanup may be needed.", e)

    @staticmethod
    def _to_uuid(key: str) -> str:
        """Deterministic UUID from any string key."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    @staticmethod
    def _format_point(point) -> dict[str, Any]:
        """Format a raw Qdrant point (no score) into a clean dict."""
        return {
            "id": str(point.id),
            "text": point.payload.get("text"),
            "metadata": {k: v for k, v in point.payload.items() if k != "text"},
        }

    @staticmethod
    def _format_scored(point) -> dict[str, Any]:
        """Format a scored search result into a clean dict."""
        return {
            "id": str(point.id),
            "text": point.payload.get("text"),
            "score": round(point.score, 4),
            "metadata": {k: v for k, v in point.payload.items() if k != "text"},
        }
