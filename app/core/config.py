from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "rag_app"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"
    API_KEY: str | None = None
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "https://depi-rag-frontend.vercel.app",
    ]
    CORS_ORIGIN_REGEX: str = r"http://(localhost|127\.0\.0\.1):517\d"
    
    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    ENABLE_PII_REDACTION: bool = True

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
    QDRANT_API_KEY: str | None = None
    QDRANT_PATH: str = "./qdrant_storage"       # used only when QDRANT_REMOTE=False
    QDRANT_HTTPS: bool = False

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
    # RAG inference
    # ------------------------------------------------------------------
    RAG_RETRIEVAL_TOP_K: int = 10
    RAG_RERANK_TOP_K: int = 5

    # ------------------------------------------------------------------
    # Metadata and artifact storage
    # ------------------------------------------------------------------
    METADATA_BACKEND: str = "sqlite"  # "json", "sqlite", or "azure_sql"
    METADATA_DB_PATH: str = "uploads/app_metadata.sqlite3"
    AZURE_SQL_CONNECTION_STRING: str | None = None
    ARTIFACT_STORAGE_BACKEND: str = "local"  # "local" or "azure_blob"
    AZURE_STORAGE_CONNECTION_STRING: str | None = None
    AZURE_BLOB_CONTAINER: str = "rag-artifacts"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: str = "cohere"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 400

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_api_version: str = "2024-02-01"

    cohere_api_key: str | None = None
    cohere_chat_model: str = "command-a-03-2025"

    # ------------------------------------------------------------------
    # Optional reranking
    # ------------------------------------------------------------------
    reranker_provider: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    azure_cohere_base_url: str | None = None
    azure_cohere_api_key: str | None = None
    azure_cohere_model: str = "model"


settings = Settings()
