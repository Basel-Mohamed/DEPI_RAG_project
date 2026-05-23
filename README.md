# DEPI RAG Customer Support Chatbot

Production-oriented FastAPI RAG service for customer support automation. It ingests user-provided PDFs, preprocesses them into chunks, builds dense/sparse Qdrant indexes, answers questions with a configured LLM provider, stores metadata in SQLite by default, can mirror artifacts to MinIO, exposes Grafana/Prometheus-friendly monitoring metrics, supports API-key auth, and includes Azure App Service deployment assets.

## Local Run

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
docker compose -f docker/docker-compose.yml up -d qdrant
.\venv\Scripts\python.exe -m uvicorn main:app --reload
```

For local development, set `AUTH_ENABLED=False`. For deployment, set `DEPLOYMENT_ENV=production`, `AUTH_ENABLED=True`, and a long random `API_KEY`. Production startup fails if auth is not configured.

## Main APIs

- `GET /health` public Azure health probe.
- `POST /files` upload a PDF.
- `POST /files/build?file_id=...` build embeddings/vector index.
- `GET /files` and `GET /files?file_id=...` inspect upload/build status.
- `DELETE /files?file_id=...` delete uploaded file and vectors.
- `POST /ask` ask a RAG question.
- `POST /ask/stream` stream NDJSON answer chunks.
- `GET /monitoring/health`, `/monitoring/metrics`, `/monitoring/rag-summary` operational monitoring.
- `GET /monitoring/prometheus` Prometheus text format for Grafana dashboards.

Protected endpoints require:

```text
X-API-Key: <API_KEY>
```

## Storage And Jobs

- Metadata uses SQLite by default: `METADATA_BACKEND=sqlite`, `METADATA_DB_PATH=uploads/app_metadata.sqlite3`.
- `METADATA_BACKEND=json` is still available for tiny local experiments.
- Azure Blob artifact mirroring is enabled with `ARTIFACT_STORAGE_BACKEND=azure_blob`,
  `AZURE_STORAGE_CONNECTION_STRING`, and `AZURE_BLOB_CONTAINER`.
- MinIO/S3-compatible artifact mirroring is still available for local/self-hosted setups with
  `ARTIFACT_STORAGE_BACKEND=minio` plus `MINIO_*` settings.
- The current build job model is intentionally simple for one-admin operation: upload records metadata, build marks the file `building`, FastAPI runs one in-app background build, and SQLite persists the status.
- For multi-admin or high-volume production, replace the in-app background job with a real queue/worker.

## Azure Deployment

Target: Azure App Service for Containers + Azure Container Registry + Qdrant Cloud.

1. Create Qdrant Cloud cluster and collection credentials.
2. Create Azure Container Registry and App Service for Containers.
3. Configure App Service settings from `.env.example`, especially `AUTH_ENABLED=True`, `API_KEY`, `QDRANT_REMOTE=True`, `QDRANT_HTTPS=True`, `QDRANT_HOST`, `QDRANT_PORT=6333`, `QDRANT_API_KEY`, LLM provider credentials, and Azure Blob artifact settings.
4. Add GitHub secrets used by `.github/workflows/azure-app-service.yml`.
5. Push to `main` or run the workflow manually.

Detailed deployment notes are in `docs/azure-deployment.md`. For self-hosted storage instead of Azure Blob, use MinIO from Docker Compose locally or deploy a managed S3-compatible service separately.

## Evaluation, Monitoring, Refresh

Run evaluation:

```powershell
python scripts/evaluate_rag.py --data data/evaluation/questions.json --output-dir reports/evaluation
```

Refresh embeddings for uploaded documents:

```powershell
python scripts/refresh_embeddings.py
```

Milestone docs live under `docs/`:

- `preprocessing-pipeline.md`
- `eda-methodology.md`
- `evaluation.md`
- `integration-security.md`
- `monitoring-retraining.md`
- `final-report.md`
- `demo-presentation.md`
- `business-kpi-impact.md`

## Production Notes

The current one-admin deployment path uses SQLite plus local/MinIO artifacts. For multi-instance production, move metadata to PostgreSQL and use MinIO or Azure Blob as the shared artifact store. Grafana can scrape `/monitoring/prometheus` through Prometheus; Azure Monitor/Application Insights remains the recommended Azure-native option.
