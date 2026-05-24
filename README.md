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

## Supported Document Upload Schemas

The upload endpoint accepts these file types:

```text
.pdf, .csv, .json, .txt, .md
```

Unsupported file extensions are rejected with `415 Unsupported Media Type`. Duplicate uploads are rejected with `409 Conflict` when either the original filename already exists or the uploaded file content matches an existing document by SHA-256 hash.

### PDF

PDF files are extracted page by page with Docling. There is no required schema inside the PDF.

Each extracted page becomes a page record before chunking:

```json
{
  "text": "Extracted page text",
  "page_number": 1,
  "metadata": {}
}
```

### CSV

CSV files may use comma or semicolon delimiters. The loader tries comma first, then falls back to semicolon.

Each CSV row becomes one page before chunking. The loader uses whichever of these text columns are present:

```text
subject, body, description, text, content, message
```

All present text columns are concatenated with a space separator. All other columns are preserved as metadata.

Example:

```csv
subject,body,ticket_id,customer_type
Login issue,User cannot reset password,TCK-001,premium
Billing question,Invoice total is incorrect,TCK-002,standard
```

The first row becomes:

```json
{
  "text": "Login issue User cannot reset password",
  "page_number": 1,
  "metadata": {
    "ticket_id": "TCK-001",
    "customer_type": "premium"
  }
}
```

If none of the expected text columns are present, rows still load, but their text will be empty and no chunks will be produced from those rows.

### JSON

JSON files can contain either a single object or an array of objects.

Default text fields:

```text
question -> optional title
answer   -> main text
```

If `question` is present, it is prepended to `answer` with a newline. All other keys are preserved as metadata.

Example array:

```json
[
  {
    "question": "How do I reset my password?",
    "answer": "Click forgot password and follow the email link.",
    "category": "account",
    "priority": "low"
  }
]
```

The first item becomes:

```json
{
  "text": "How do I reset my password?\nClick forgot password and follow the email link.",
  "page_number": 1,
  "metadata": {
    "category": "account",
    "priority": "low"
  }
}
```

Example single object:

```json
{
  "question": "Where can I find invoices?",
  "answer": "Invoices are available from the billing dashboard.",
  "category": "billing"
}
```

If `answer` is missing, the item still loads, but only `question` contributes text when present.

### TXT and Markdown

`.txt` and `.md` files are read as plain text. The entire file becomes one page before chunking.

Example page record:

```json
{
  "text": "Full file contents...",
  "page_number": 1,
  "metadata": {}
}
```

### Chunk Metadata

After loading, every supported file type is split into chunks. Each chunk contains:

```json
{
  "text": "Chunk text after whitespace cleanup and optional PII redaction.",
  "metadata": {
    "source": "uploads/file-id.ext",
    "page_number": 1,
    "chunk_index": 0
  }
}
```

CSV and JSON metadata fields are merged into the chunk metadata before `source`, `page_number`, and `chunk_index` are added.

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

Uploaded documents are stored locally by default. To persist uploaded files in Azure Blob Storage, set:

```env
ARTIFACT_STORAGE_BACKEND=azure_blob
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=...
AZURE_BLOB_CONTAINER=rag-artifacts
```

The API keeps a local working copy for document processing and stores the durable artifact URI in the configured metadata backend.

## Metadata Storage

File registry metadata and feedback records are stored in SQLite by default:

```env
METADATA_BACKEND=sqlite
METADATA_DB_PATH=uploads/app_metadata.sqlite3
```

For Azure SQL, install the ODBC driver on the host and set:

```env
METADATA_BACKEND=azure_sql
AZURE_SQL_CONNECTION_STRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:<server>.database.windows.net,1433;Database=<database>;Uid=<user>;Pwd=<password>;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
```

Use `METADATA_BACKEND=json` only when you need the legacy `uploads/files.json` and `feedback/feedback.json` files.

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
QDRANT_API_KEY=
```

Inside Docker Compose, the API service overrides `QDRANT_HOST` to `qdrant`, which is the service name on the Compose network.

Qdrant REST UI/API:

```text
http://localhost:6333/dashboard
```

## React Demo Frontend

The demo UI lives in `frontend/` and authenticates against `GET /auth/verify` with the same `X-API-Key` value used by the protected API routes.

Start the backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --reload
```

Start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend defaults to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` if the backend is running somewhere else.
