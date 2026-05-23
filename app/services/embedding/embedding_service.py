import logging
import time
from typing import Any

from fastembed import TextEmbedding
from app.core.config import Settings, settings as global_settings
from app.controllers.monitoring_controller import MonitoringMetrics

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-large"


class EmbeddingService:
    """Generate dense embeddings for text chunks using fastembed.

    Usage:
        service = EmbeddingService(settings)
        embedded_chunks = service.embed_chunks(chunks)

        # Each chunk gets an added "embedding" key:
        # {"text": ..., "metadata": ..., "embedding": [0.12, -0.03, ...]}
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or global_settings
        self.model_name = settings.EMBEDDING_MODEL
        logger.info("Loading embedding model: %s", self.model_name)
        self.model = TextEmbedding(model_name=self.model_name)

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add an 'embedding' field to each chunk. Returns a new list."""
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        start = time.perf_counter()
        embeddings = list(self.model.embed(texts))  # materialise the generator
        MonitoringMetrics.record_embedding_latency(
            (time.perf_counter() - start) * 1000
        )
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} vectors for {len(chunks)} chunks."
            )

        logger.info("Embedded %d chunks with model '%s'.", len(chunks), self.model_name)

        return [
            {**chunk, "embedding": embedding.tolist()}
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        start = time.perf_counter()
        embedding = list(self.model.query_embed(query))[0].tolist()
        MonitoringMetrics.record_embedding_latency(
            (time.perf_counter() - start) * 1000
        )
        return embedding
