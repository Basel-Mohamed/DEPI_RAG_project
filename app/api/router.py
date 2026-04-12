from fastapi import APIRouter

from app.api.v1.routes.documents import router as documents_router
from app.api.v1.routes.query import router as query_router
from app.core.config import get_settings


settings = get_settings()

api_router = APIRouter()
v1_router = APIRouter(prefix=settings.api_v1_prefix)

v1_router.include_router(documents_router, prefix="/documents", tags=["documents"])
v1_router.include_router(query_router, prefix="/query", tags=["query"])

api_router.include_router(v1_router)

