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
METADATA_BACKEND=azure_sql
AZURE_SQL_CONNECTION_STRING=<azure-sql-odbc-connection-string>
ARTIFACT_STORAGE_BACKEND=azure_blob
AZURE_STORAGE_CONNECTION_STRING=<storage-account-connection-string>
AZURE_BLOB_CONTAINER=rag-artifacts
```

GitHub repository secrets:

- `AZURE_CREDENTIALS`
- `ACR_NAME`
- `AZURE_RESOURCE_GROUP`
- `AZURE_WEBAPP_NAME`

Health probe:

- Use `GET /health`.
- Keep `/health` unauthenticated.

Azure Blob artifact storage:

1. Create an Azure Storage Account.
2. Create a private Blob container, for example `rag-artifacts`.
3. Copy the Storage Account connection string from **Security + networking → Access keys**.
4. Add `ARTIFACT_STORAGE_BACKEND=azure_blob`, `AZURE_STORAGE_CONNECTION_STRING`, and
   `AZURE_BLOB_CONTAINER=rag-artifacts` to App Service application settings.
5. Restart the App Service.

Azure SQL metadata storage:

1. Create an Azure SQL Database.
2. In the SQL Database resource, open **Settings → Connection strings** and copy the ODBC connection string.
3. Replace `{your_password}` in the string with the SQL admin password you set during database creation.
4. Add `METADATA_BACKEND=azure_sql` and `AZURE_SQL_CONNECTION_STRING` to App Service application settings.
5. In the SQL Server networking/firewall settings, allow the App Service outbound IPs or temporarily enable Azure services access for testing.
6. Restart the App Service.

Expected connection string shape:

```text
Driver={ODBC Driver 18 for SQL Server};Server=tcp:<server>.database.windows.net,1433;Database=<database>;Uid=<user>;Pwd=<password>;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
```

Production upgrade path:

- Use Azure Blob Storage for uploaded source files and other durable artifacts.
- Use Azure SQL for metadata when App Service storage is not enough or when you plan to run more than one API instance.
- MinIO is a valid S3-compatible artifact store if you do not want Azure Blob Storage.
- Send logs/metrics to Application Insights or Azure Monitor.
- Use Azure Key Vault references for API keys and provider credentials.
