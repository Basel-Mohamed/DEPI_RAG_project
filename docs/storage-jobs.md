# Storage And Jobs

Recommended one-admin setup:

- SQLite for upload/build metadata.
- Local filesystem or MinIO for PDFs and generated artifacts.
- In-app FastAPI background build jobs.

Why this is acceptable for one admin:

- Only one person is creating builds.
- SQLite persists build/upload state across restarts.
- Qdrant remains the source of truth for vectors.

Limitations:

- In-app background jobs can be interrupted by restarts.
- SQLite is not ideal for many concurrent writers or multiple app instances.
- Local filesystem is not shared across replicas.

Upgrade path:

- PostgreSQL for metadata.
- MinIO or Azure Blob for artifacts.
- Queue-backed worker for builds/rebuilds.
