# Customer Support RAG-Powered Intelligent Chatbot

## Executive Summary

This graduation project delivers a production-oriented Retrieval-Augmented Generation support chatbot. It ingests business support documents, preprocesses and redacts text, stores dense and sparse vectors in Qdrant, answers user questions through a grounded RAG pipeline, records user feedback, exposes Prometheus-ready metrics, and supports scheduled re-indexing through MLflow-tracked jobs.

## Problem Statement

Support teams spend significant time searching manuals, FAQs, policies, and historical tickets. Traditional keyword search often misses paraphrased questions, while standalone LLM answers can be ungrounded. The project solves this by combining vector retrieval, hybrid keyword matching, source-grounded generation, and operational monitoring.

## Implemented Architecture

1. Data ingestion accepts PDF, CSV, JSON, TXT, and Markdown files.
2. Preprocessing extracts page records, cleans whitespace, redacts PII, chunks text, and removes duplicate chunks.
3. Embedding uses `intfloat/multilingual-e5-large`; sparse retrieval uses FastEmbed BM25.
4. Qdrant stores dense and sparse vectors with source, page, and chunk metadata.
5. Inference retrieves top documents, optionally reranks them, and generates a source-backed answer.
6. FastAPI exposes upload, build, ask, streaming ask, feedback, metrics, and health endpoints.
7. Monitoring records latency, token usage, Qdrant latency, feedback volume, and satisfaction score.
8. MLOps support is provided through MLflow experiment tracking and a scheduled reindex script.

## Milestone Coverage

| Milestone | Requirement | Current implementation |
| --- | --- | --- |
| 1 | Data collection and preprocessing | Corpus builder, loaders, chunking, PII redaction, deduplication, EDA report |
| 2 | RAG model development and evaluation | Qdrant hybrid retrieval, Cohere/Azure generation, optional reranking, MLflow comparison |
| 3 | Azure deployment and integration | Dockerized API, API-key auth, Azure Blob and Azure SQL configuration hooks |
| 4 | MLOps and monitoring | Prometheus export, Grafana dashboard JSON, feedback KPI, MLflow reindex pipeline |
| 5 | Final documentation and presentation | Final report, demo deck, KPI impact analysis, integration/security docs |

## Evaluation Summary

The current MLflow comparison shows strong retrieval behavior on the available evaluation example. Context recall and context precision are 1.0000 for the recorded runs. Hybrid retrieval achieved faithfulness of 0.8000 with answer relevancy, context recall, and context precision all at 1.0000. The evaluation set should be expanded before production sign-off.

## Security And Governance

The project includes API key protection, PII redaction before indexing, duplicate upload detection, request-level logging with IDs, and configurable metadata/artifact backends. For production Azure deployment, API keys should be managed through Key Vault, monitoring should be private-network restricted, and Azure AD should be considered for portal users.

## Business Value

The chatbot reduces repetitive support lookup effort, improves consistency of policy answers, provides source traceability, and creates a measurable feedback loop. The strongest KPIs are first response time, deflection rate, agent handle time, satisfaction score, and knowledge freshness.

## Future Enhancements

- Add a larger support-ticket and FAQ dataset.
- Add BLEU and ROUGE for requirement completeness while keeping RAGAS metrics as the main quality signal.
- Add Azure Application Insights traces and alert rules.
- Add role-based access for admin-only ingestion routes.
- Add automated negative-feedback review into the reindex and evaluation loop.
