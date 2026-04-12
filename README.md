# DEPI RAG Backend

Backend-only starter for a Retrieval-Augmented Generation project organized with an MVC-inspired structure.

## Architecture

This repository is intentionally scoped to backend and model concerns:

- `app/api`: FastAPI route registration
- `app/controllers`: request orchestration layer
- `app/services`: business logic for ingestion and retrieval/generation
- `app/repositories`: data access and vector-store abstractions
- `app/models`: domain models
- `app/schemas`: request and response contracts
- `app/core`: configuration and shared app setup

The flow is:

`Route -> Controller -> Service -> Repository -> Model`

## Quick Start

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env`.
3. Start the API:

```bash
uvicorn app.main:app --reload
```

## Available Endpoints

- `GET /health`
- `POST /api/v1/documents`
- `POST /api/v1/query`

## Notes

- The current vector store and LLM integrations are in-memory stubs so the project is runnable from day one.
- You can later replace the repository and provider implementations without changing the API shape.

