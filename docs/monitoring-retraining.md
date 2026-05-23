# Monitoring And Retraining

Monitoring endpoints:

- `GET /monitoring/health`
- `GET /monitoring/metrics`
- `GET /monitoring/rag-summary`
- `GET /monitoring/prometheus`

Tracked in-process metrics:

- HTTP request and error counts.
- Upload and build counts.
- Build failures.
- Inference requests and failures.
- Retrieved document counts.
- Fallback answer counts.
- HTTP and inference latency summaries.

The in-process registry is demo-grade. For production, export logs and metrics to Azure Monitor/Application Insights.

Grafana option:

- Run Prometheus and Grafana from `docker/docker-compose.yml`.
- Prometheus scrapes `/monitoring/prometheus`.
- Grafana connects to Prometheus at `http://prometheus:9090`.

Refresh mechanism:

```powershell
python scripts/refresh_embeddings.py
```

This rebuilds embeddings and vector indexes from already uploaded documents. It does not fine-tune model weights.

Scheduling options:

- Azure WebJobs triggered on a schedule.
- GitHub Actions cron that runs the refresh command against the deployed environment.
- Manual Azure SSH/console invocation for demos.
