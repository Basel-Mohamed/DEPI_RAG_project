# Monitoring Setup

## Overview

The RAG backend records operational metrics for request latency, embedding latency, reranking latency, LLM latency, Qdrant latency, LLM token usage, and user feedback. Metrics are persisted through the metadata store: SQLite by default at `uploads/app_metadata.sqlite3`, and Azure SQL in production when `METADATA_BACKEND=azure_sql`.

The flow is:

1. FastAPI controllers and services record metrics through `MonitoringMetrics`.
2. The metadata store persists records in `monitoring_metrics` and `feedback`.
3. `/metrics` returns an application JSON summary.
4. `/metrics/prometheus` exports Prometheus text format.
5. Prometheus scrapes that endpoint, and Grafana visualizes the scraped series.

## Metrics Reference

| Metric name | Type | Description | Unit |
| --- | --- | --- | --- |
| `rag_request_total` | Counter | Total number of inference requests | requests |
| `rag_request_latency_ms_avg` | Gauge | Average end-to-end request latency | milliseconds |
| `rag_embedding_latency_ms_avg` | Gauge | Average embedding latency | milliseconds |
| `rag_reranking_latency_ms_avg` | Gauge | Average reranking latency | milliseconds |
| `rag_llm_latency_ms_avg` | Gauge | Average LLM generation latency | milliseconds |
| `rag_qdrant_latency_ms_avg` | Gauge | Average Qdrant operation latency | milliseconds |
| `rag_llm_tokens_total` | Counter | Total generated LLM tokens | tokens |
| `rag_feedback_total` | Counter | Total feedback submissions | submissions |
| `rag_feedback_positive_total` | Counter | Positive feedback submissions | submissions |
| `rag_satisfaction_score` | Gauge | Positive feedback divided by total feedback | ratio |

## Prometheus Scrape Setup

Add a scrape job to `prometheus.yml`. Use the same API key configured for the application.

```yaml
scrape_configs:
  - job_name: rag_backend
    metrics_path: /metrics/prometheus
    scheme: http
    static_configs:
      - targets:
          - localhost:8000
    authorization:
      type: Bearer
      credentials: ${API_KEY}
```

If your app expects `X-API-Key` instead of bearer auth, configure Prometheus with a reverse proxy that maps the bearer token into the `X-API-Key` header, or disable auth only on an internal metrics-only deployment.

## Grafana Dashboard Import

1. Generate the dashboard JSON:

```bash
python scripts/generate_grafana_dashboard.py
```

2. Open Grafana.
3. Go to Dashboards -> Import -> Upload JSON file.
4. Upload `monitoring/grafana_dashboard.json`.
5. Select the Prometheus data source and import.

[SCREENSHOT: Dashboard import dialog]

## Running The Monitoring Notebook

Run:

```bash
jupyter notebook notebooks/03_monitoring.ipynb
```

The notebook first tries to call `http://localhost:8000/metrics` with `X-API-Key` from `.env`. If the API is offline or unreachable, it falls back to reading `uploads/app_metadata.sqlite3` directly.

## MLflow UI

Run:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open the `RAG Pipeline Comparison` experiment to compare retrieval modes and reranker settings. The key metrics to inspect are faithfulness, answer relevancy, context recall, and context precision.

## Reindex Pipeline

Run once:

```bash
python scripts/reindex_pipeline.py --mode once
```

Run on a 24-hour schedule:

```bash
python scripts/reindex_pipeline.py --mode schedule --interval-hours 24
```

Each reindex run logs parameters, counts, duration, and a `reindex_summary.json` artifact to the `Reindex Pipeline` MLflow experiment.

## Alerting Guidance

Recommended thresholds to watch:

| Signal | Suggested alert |
| --- | --- |
| Request latency | p95 greater than 3000 ms |
| Satisfaction score | Below 0.6 |
| Reindex failure rate | Greater than 10% |
