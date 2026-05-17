from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.core.metrics import REGISTRY, QDRANT_UP
from app.schemas.monitoring import HealthResponse, StatsResponse, MetricSample
from app.services.monitoring.grafana_service import GrafanaService


class MonitoringController:

    def __init__(
        self,
        grafana_url: str,
        prometheus_url: str,
        qdrant_url: str,
    ) -> None:
        self.grafana_service = GrafanaService(
            grafana_url=grafana_url,
            prometheus_url=prometheus_url,
        )
        self.qdrant_url = qdrant_url

    # ── /metrics ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_metrics() -> tuple[bytes, str]:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

    # ── /health ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_health(environment: str | None = None) -> HealthResponse:
        return HealthResponse(
            status="ok",
            environment=environment,
        )

    # ── /monitoring/stats ────────────────────────────────────────────────────

    async def get_stats(self) -> StatsResponse:
        await self.grafana_service.check_qdrant(self.qdrant_url)

        collected = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                collected[sample.name] = MetricSample(
                    value=sample.value,
                    labels=dict(sample.labels),
                )

        rag_prefixes = [
            "rag_", "llm_", "embedding_",
            "reranking_", "qdrant_", "document_",
        ]
        rag_stats = {
            key: val
            for key, val in collected.items()
            if any(key.startswith(prefix) for prefix in rag_prefixes)
        }

        return StatsResponse(
            qdrant_up=bool(QDRANT_UP._value.get()),
            metrics=rag_stats,
        )