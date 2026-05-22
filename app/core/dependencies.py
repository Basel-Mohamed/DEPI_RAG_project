import logging
from functools import lru_cache

from app.services.rag.rag_builder import BuildService
from app.services.rag.rag_inference import RagInferencePipeline

logger = logging.getLogger(__name__)


@lru_cache
def get_embedding_service():
    from app.services.embedding.embedding_service import EmbeddingService

    logger.info("initializing embedding service")
    return EmbeddingService()


@lru_cache
def get_document_processor():
    from app.services.preprocessing.preprocessing_service import DocumentProcessor

    logger.info("initializing document processor")
    return DocumentProcessor()


@lru_cache
def get_qdrant_service():
    from app.services.vectorstore.qdrant_store import QdrantService

    logger.info("initializing qdrant service")
    return QdrantService(embedding_service=get_embedding_service())


@lru_cache
def get_build_service() -> BuildService:
    logger.info("initializing build service")
    return BuildService(
        document_processor=get_document_processor(),
        embedding_service=get_embedding_service(),
        vector_store=get_qdrant_service(),
    )


@lru_cache
def get_inference_pipeline() -> RagInferencePipeline:
    logger.info("initializing inference pipeline")
    return RagInferencePipeline(vector_store=get_qdrant_service())
