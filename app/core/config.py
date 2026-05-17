from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str
    APP_VERSION: str

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------
    IMAGE_SCALE: float = 2.0
    IMAGE_FORMAT: str = "PNG"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    SPARSE_MODEL: str = "Qdrant/bm25"           # BM25 sparse model via fastembed

    # ------------------------------------------------------------------
    # Qdrant connection
    # ------------------------------------------------------------------
    QDRANT_REMOTE: bool = False                 # True → remote, False → local file
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_PATH: str = "./qdrant_storage"       # used only when QDRANT_REMOTE=False

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    COLLECTION_NAME: str = "documents"
    SCORE_THRESHOLD: float = 0.6
    DENSE_VECTOR_SIZE: int = 1024               # multilingual-e5-large output dim
    DENSE_ON_DISK: bool = False
    SPARSE_ON_DISK: bool = False

    # Search mode: "dense" | "sparse" | "hybrid"
    RETRIEVAL_MODE: str = "hybrid"

    # ------------------------------------------------------------------
    # HNSW index
    # ------------------------------------------------------------------
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCT: int = 100
    HNSW_FULL_SCAN_THRESHOLD: int = 10_000
    HNSW_ON_DISK: bool = False

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    OPTIMIZATION_INDEXING_THRESHOLD: int = 20_000
    OPTIMIZATION_MEMMAP_THRESHOLD: int = 50_000

    # ------------------------------------------------------------------
    # Scalar quantization (INT8)
    # ------------------------------------------------------------------
    QUANTIZATION_QUANTILE: float = 0.99
    QUANTIZATION_ALWAYS_RAM: bool = True

    # ------------------------------------------------------------------
    # Environment & Monitoring
    # ------------------------------------------------------------------
    ENVIRONMENT: str = "development"
    PROMETHEUS_URL: str = "http://localhost:9090"
    GRAFANA_URL: str = "http://localhost:3000"
    QDRANT_URL: str = "http://localhost:6333"


settings = Settings()