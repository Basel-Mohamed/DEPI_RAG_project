# Final Report

## Overview

The system is a customer support RAG chatbot that ingests support documents, indexes them in Qdrant, and answers user questions with cited retrieved context.

## Completed Capabilities

- PDF ingestion and preprocessing.
- Dense/sparse/hybrid retrieval.
- Optional reranking.
- LLM answer generation and streaming.
- API-key security.
- Dockerized Azure App Service deployment path.
- Qdrant Cloud production vector backend support.
- Monitoring endpoints and request logging.
- Evaluation script and local MLflow tracking.
- Embedding refresh script.

## Known Limitations

- Support tickets, FAQs, and manuals are user-provided; no live third-party connectors are included.
- App Service filesystem storage is demo-grade.
- Metrics are in-process and reset on restart.
- Retraining means index refresh, not model-weight fine-tuning.

## Recommended Next Steps

- Add Azure Blob Storage for artifacts.
- Add Application Insights export.
- Add Azure AD auth for enterprise users.
- Add real support-system connectors.
