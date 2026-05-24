import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.controllers.feedback_controller import FeedbackController
from app.controllers.monitoring_controller import MonitoringController
from app.core.auth import verify_api_key
from app.core.dependencies import get_qdrant_service
from app.services.monitoring.grafana_service import (
    GrafanaDashboardBuilder,
    PrometheusExporter,
)
from app.schemas.monitoring import (
    MetricsResetResponse,
    MetricsResponse,
    QdrantHealthResponse,
)

router = APIRouter(tags=["monitoring"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)


def get_monitoring_controller(
    qdrant_service: Any = Depends(get_qdrant_service),
) -> MonitoringController:
    return MonitoringController(qdrant_service)


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    request: Request,
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    request.state.skip_monitoring_metrics = True
    logger.info("metrics endpoint received")
    try:
        return controller.metrics_summary()
    except Exception:
        logger.exception("metrics endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics lookup failed. Check server logs for the request id.",
        )


@router.post("/metrics/reset", response_model=MetricsResetResponse)
def reset_metrics(
    request: Request,
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    request.state.skip_monitoring_metrics = True
    logger.info("metrics reset endpoint received")
    try:
        return controller.reset_metrics()
    except Exception:
        logger.exception("metrics reset endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics reset failed. Check server logs for the request id.",
        )


@router.get("/health/qdrant", response_model=QdrantHealthResponse)
def get_qdrant_health(
    request: Request,
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    request.state.skip_monitoring_metrics = True
    logger.info("qdrant health endpoint received")
    try:
        return controller.qdrant_health()
    except Exception:
        logger.exception("qdrant health endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Qdrant health lookup failed. Check server logs for the request id.",
        )


@router.get("/metrics/prometheus")
def get_prometheus_metrics(
    request: Request,
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> Response:
    request.state.skip_monitoring_metrics = True
    logger.info("prometheus metrics endpoint received")
    try:
        snapshot = controller.metrics.snapshot()
        feedback_controller: FeedbackController = controller.feedback_controller
        feedback_summary = feedback_controller.satisfaction_summary()
        prometheus_text = PrometheusExporter().export(snapshot, feedback_summary)
        return Response(
            content=prometheus_text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except Exception:
        logger.exception("prometheus metrics endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prometheus metrics export failed. Check server logs for the request id.",
        )


@router.get("/dashboard/grafana")
def get_grafana_dashboard(
    request: Request,
    controller: MonitoringController = Depends(get_monitoring_controller),
) -> dict:
    request.state.skip_monitoring_metrics = True
    logger.info("grafana dashboard endpoint received")
    try:
        return GrafanaDashboardBuilder().build()
    except Exception:
        logger.exception("grafana dashboard endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grafana dashboard generation failed. Check server logs for the request id.",
        )
