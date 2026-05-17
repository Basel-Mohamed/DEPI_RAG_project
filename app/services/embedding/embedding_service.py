import time
import logging
from typing import Any

from fastembed import TextEmbedding
from app.core.config import Settings, settings as global_settings
from app.core.metrics import EMBEDDING_LATENCY, CHUNKS_INGESTED

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

        # ── measure embedding latency ────────────────────────────────────────
        start = time.perf_counter()
        embeddings = list(self.model.embed(texts))  # materialise the generator
        EMBEDDING_LATENCY.observe(time.perf_counter() - start)
        CHUNKS_INGESTED.inc(len(chunks))
        # ────────────────────────────────────────────────────────────────────

        logger.info("Embedded %d chunks with model '%s'.", len(chunks), self.model_name)

        return [
            {**chunk, "embedding": embedding.tolist()}
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""

        # ── measure query embedding latency ──────────────────────────────────
        start = time.perf_counter()
        result = list(self.model.query_embed(query))[0].tolist()
        EMBEDDING_LATENCY.observe(time.perf_counter() - start)
        # ────────────────────────────────────────────────────────────────────

        return result