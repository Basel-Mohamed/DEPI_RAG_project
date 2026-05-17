from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.controllers.monitoring_controller import MonitoringController
from app.schemas.monitoring import HealthResponse, StatsResponse
from app.core.config import settings

router = APIRouter()

controller = MonitoringController(
    grafana_url=settings.GRAFANA_URL,
    prometheus_url=settings.PROMETHEUS_URL,
    qdrant_url=settings.QDRANT_URL,
)


# ── Endpoint 1: Prometheus scrape ────────────────────────────────────────────

@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    data, content_type = controller.get_metrics()
    return PlainTextResponse(content=data, media_type=content_type)


# ── Endpoint 2: Liveness probe ───────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return controller.get_health(environment=settings.ENVIRONMENT)


# ── Endpoint 3: Human-readable stats snapshot ────────────────────────────────

@router.get("/monitoring/stats", response_model=StatsResponse)
async def stats():
    return await controller.get_stats()