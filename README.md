# DEPI RAG Project

## API Authentication

Protected API routes require an `X-API-Key` header matching the `API_KEY` environment variable.

```env
API_KEY=replace-with-a-secret-value
```

Example request:

```powershell
curl.exe -H "X-API-Key: replace-with-a-secret-value" http://localhost:8000/files
```

## User Feedback

Use `POST /feedback` to persist answer-level feedback for KPI reporting.

```powershell
curl.exe -X POST http://localhost:8000/feedback `
  -H "X-API-Key: replace-with-a-secret-value" `
  -H "Content-Type: application/json" `
  -d "{\"session_id\":\"session-123\",\"question\":\"What is the refund policy?\",\"answer\":\"Refunds are available within 30 days.\",\"rating\":1,\"timestamp\":\"2026-05-23T15:30:00Z\"}"
```

`rating` must be `1` for positive feedback or `-1` for negative feedback.

List saved feedback:

```powershell
curl.exe -H "X-API-Key: replace-with-a-secret-value" http://localhost:8000/feedback
```

List feedback for one session:

```powershell
curl.exe -H "X-API-Key: replace-with-a-secret-value" "http://localhost:8000/feedback?session_id=session-123&limit=10"
```

Get the satisfaction KPI:

```powershell
curl.exe -H "X-API-Key: replace-with-a-secret-value" http://localhost:8000/feedback/satisfaction
```

## Azure Blob Artifact Storage

Uploaded PDFs are stored locally by default. To persist uploaded files in Azure Blob Storage, set:

```env
ARTIFACT_STORAGE_BACKEND=azure_blob
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=...
AZURE_BLOB_CONTAINER=rag-artifacts
```

The API keeps a local working copy for document processing and stores the durable artifact URI in `uploads/files.json`.

## Local Qdrant with Docker

This project is configured to use a local Qdrant server over HTTP.

Start Qdrant and the API together:

```powershell
docker compose -f docker/docker-compose.yml up -d --build
```

Run only Qdrant, then start the API from your local Python environment:

```powershell
docker compose -f docker/docker-compose.yml up -d qdrant
.\venv\Scripts\python.exe -m uvicorn main:app --reload
```

The `.env` file points local Python runs at Qdrant on `localhost:6333`:

```env
QDRANT_REMOTE=True
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Inside Docker Compose, the API service overrides `QDRANT_HOST` to `qdrant`, which is the service name on the Compose network.

Qdrant REST UI/API:

```text
http://localhost:6333/dashboard
```
