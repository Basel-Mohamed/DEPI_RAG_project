from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

REGISTRY = CollectorRegistry(auto_describe=True)

# ── Request level ────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    registry=REGISTRY,
)

# ── RAG pipeline stages ──────────────────────────────────────────────────────

EMBEDDING_LATENCY = Histogram(
    "embedding_duration_seconds",
    "Latency of Cohere embedding calls",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2],
    registry=REGISTRY,
)

RERANKING_LATENCY = Histogram(
    "reranking_duration_seconds",
    "Latency of Cohere reranking calls",
    buckets=[0.05, 0.1, 0.25, 0.5, 1],
    registry=REGISTRY,
)

LLM_LATENCY = Histogram(
    "llm_response_duration_seconds",
    "Latency of Cohere LLM response generation",
    buckets=[0.5, 1, 2, 5, 10, 30],
    registry=REGISTRY,
)

QDRANT_LATENCY = Histogram(
    "qdrant_query_duration_seconds",
    "Latency of Qdrant vector similarity search",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY,
)

# ── Counters ─────────────────────────────────────────────────────────────────

LLM_TOKENS = Counter(
    "llm_tokens_used_total",
    "Total Cohere tokens consumed",
    ["type"],          # label values: "prompt" or "completion"
    registry=REGISTRY,
)

CHUNKS_INGESTED = Counter(
    "document_chunks_ingested_total",
    "Total chunks added to the vector store",
    registry=REGISTRY,
)

DOCS_INGESTED = Counter(
    "documents_ingested_total",
    "Total documents processed by the ingestion service",
    registry=REGISTRY,
)

# ── Health gauges ────────────────────────────────────────────────────────────

QDRANT_UP = Gauge(
    "qdrant_up",
    "1 if Qdrant is reachable, 0 otherwise",
    registry=REGISTRY,
)