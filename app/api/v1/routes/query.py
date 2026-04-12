from fastapi import APIRouter, Depends

from app.controllers.query_controller import QueryController
from app.dependencies import get_query_controller
from app.schemas.query import QueryRequest, QueryResponse


router = APIRouter()


@router.post("", response_model=QueryResponse)
def query_documents(
    payload: QueryRequest,
    controller: QueryController = Depends(get_query_controller),
) -> QueryResponse:
    return controller.query(payload)

