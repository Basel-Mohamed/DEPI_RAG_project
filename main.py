import logging
import time
import uuid

from fastapi import Depends, FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.build import router as build_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.inference import router as inference_router
from app.api.routes.monitoring import router as monitoring_router
from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.logging import configure_logging, request_id_context
from app.controllers.monitoring_controller import MonitoringMetrics


configure_logging()
logger = logging.getLogger(__name__)


def should_record_request_metrics(request: Request) -> bool:
    if getattr(request.state, "skip_monitoring_metrics", False):
        return False
    return request.method == "POST" and request.url.path == "/ask"


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(build_router)
app.include_router(inference_router)
app.include_router(feedback_router)
app.include_router(monitoring_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_context.set(request_id)
    start = time.perf_counter()
    logger.info("request started method=%s path=%s", request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if should_record_request_metrics(request):
            MonitoringMetrics.record_request_latency(duration_ms)
        logger.exception(
            "request failed method=%s path=%s duration_ms=%s",
            request.method,
            request.url.path,
            duration_ms,
        )
        request_id_context.reset(token)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    if should_record_request_metrics(request):
        MonitoringMetrics.record_request_latency(duration_ms)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request completed method=%s path=%s status_code=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    request_id_context.reset(token)
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/verify", dependencies=[Depends(verify_api_key)])
def verify_auth() -> dict[str, bool]:
    return {"authenticated": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
