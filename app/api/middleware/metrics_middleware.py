import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start

        REQUEST_LATENCY.labels(
            endpoint=request.url.path,
        ).observe(duration)

        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status_code=str(response.status_code),
        ).inc()

        return response