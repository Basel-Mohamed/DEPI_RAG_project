# Integration & Security Document

## Runtime Architecture

The service is exposed through FastAPI in `main.py`. It registers build, inference, feedback, and monitoring routers. Qdrant stores dense and sparse vectors. Metadata and feedback are stored in SQLite by default or Azure SQL in production. Uploaded artifacts are stored locally by default or in Azure Blob Storage when configured.

## REST API Surface

| Endpoint | Purpose | Security |
| --- | --- | --- |
| `POST /files` | Upload PDF, CSV, JSON, TXT, or Markdown support content | `X-API-Key` |
| `POST /files/build` | Process uploaded files, embed chunks, and upsert vectors | `X-API-Key` |
| `GET /files` | List uploaded/build status records | `X-API-Key` |
| `DELETE /files` | Delete files and related vectors | `X-API-Key` |
| `POST /ask` | Return a grounded RAG answer with optional sources | `X-API-Key` |
| `POST /ask/stream` | Stream answer events as NDJSON | `X-API-Key` |
| `POST /feedback` | Save answer-level user feedback | `X-API-Key` |
| `GET /feedback/satisfaction` | Return satisfaction KPI | `X-API-Key` |
| `GET /metrics` | Return JSON monitoring summary | `X-API-Key` |
| `GET /metrics/prometheus` | Export Prometheus metrics | `X-API-Key` |

## Azure Deployment Mapping

- Azure App Service or Azure Container Apps: host the FastAPI container.
- Azure Blob Storage: durable upload and artifact storage.
- Azure SQL: metadata, file registry, feedback, and monitoring persistence.
- Azure OpenAI: optional generation provider.
- Azure AI Search can be swapped in for Qdrant if the vector-store boundary is reimplemented.
- Azure Monitor and Application Insights: service logs, traces, and alerting.

## Security Controls

- API key authentication is implemented through the `X-API-Key` header.
- PII redaction is enabled before text is embedded.
- Duplicate uploads are rejected by filename and SHA-256 content hash.
- Request IDs are attached to logs and response headers for incident traceability.
- Production deployments should rotate keys through Key Vault and restrict metrics access to private networks.

## Support Portal Integration

1. The portal sends a user question to `POST /ask`.
2. The chatbot returns the answer, sources, and retrieval metadata.
3. The portal displays cited support snippets to the user or agent.
4. The portal submits thumbs-up or thumbs-down feedback to `POST /feedback`.
5. Negative feedback is reviewed and added to future evaluation and retraining batches.
