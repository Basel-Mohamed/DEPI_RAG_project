# Azure App Service Deployment

Target architecture:

- Azure App Service for Containers runs the FastAPI API image.
- Azure Container Registry stores the Docker image.
- Qdrant Cloud hosts the production vector database.
- App Service application settings hold runtime configuration and secrets.
- GitHub Actions builds, tests, pushes, and deploys the image.

Required app settings:

```env
AUTH_ENABLED=True
API_KEY=<long-random-secret>
API_KEY_HEADER=X-API-Key
DEPLOYMENT_ENV=production
QDRANT_REMOTE=True
QDRANT_HTTPS=True
QDRANT_HOST=<qdrant-cloud-host>
QDRANT_PORT=6333
QDRANT_API_KEY=<qdrant-cloud-api-key>
LLM_PROVIDER=cohere
COHERE_API_KEY=<cohere-key>
MLFLOW_TRACKING_URI=file:./mlruns
METADATA_BACKEND=sqlite
METADATA_DB_PATH=uploads/app_metadata.sqlite3
```

GitHub repository secrets:

- `AZURE_CREDENTIALS`
- `ACR_NAME`
- `AZURE_RESOURCE_GROUP`
- `AZURE_WEBAPP_NAME`

Health probe:

- Use `GET /health`.
- Keep `/health` unauthenticated.

Production upgrade path:

- Move `uploads/`, `mlruns/`, and `reports/` to Azure Blob Storage.
- For one-admin deployments, SQLite metadata is acceptable. For multi-instance production, move metadata to PostgreSQL.
- MinIO is a valid S3-compatible artifact store if you do not want Azure Blob Storage.
- Send logs/metrics to Application Insights or Azure Monitor.
- Use Azure Key Vault references for API keys and provider credentials.
