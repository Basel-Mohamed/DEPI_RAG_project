import logging
import time
import uuid

from fastapi import FastAPI
from fastapi import Request

from app.api.routes.build import router as build_router
from app.api.routes.inference import router as inference_router
from app.core.config import settings
from app.core.logging import configure_logging, request_id_context


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
app.include_router(build_router)
app.include_router(inference_router)


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
        logger.exception(
            "request failed method=%s path=%s duration_ms=%s",
            request.method,
            request.url.path,
            duration_ms,
        )
        request_id_context.reset(token)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
