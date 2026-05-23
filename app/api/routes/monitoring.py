from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app.controllers.monitoring_controller import MonitoringController
from app.core.security import require_api_key
from app.schemas.monitoring import (
    MetricsResponse,
    MonitoringHealthResponse,
    RagSummaryResponse,
)

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    dependencies=[Depends(require_api_key)],
)


def get_monitoring_controller() -> MonitoringController:
    return MonitoringController()


@router.get("/health", response_model=MonitoringHealthResponse)
def monitoring_health(
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    return controller.health()


@router.get("/metrics", response_model=MetricsResponse)
def monitoring_metrics(
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    return controller.metrics_snapshot()


@router.get("/rag-summary", response_model=RagSummaryResponse)
def monitoring_rag_summary(
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    return controller.rag_summary()


@router.get("/prometheus", response_class=PlainTextResponse)
def monitoring_prometheus(
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> str:
    return controller.prometheus()
