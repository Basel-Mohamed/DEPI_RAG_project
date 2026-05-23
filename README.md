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
