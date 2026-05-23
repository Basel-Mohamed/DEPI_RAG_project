import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.controllers.feedback_controller import FeedbackController
from app.core.auth import verify_api_key
from app.schemas.feedback import (
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSatisfactionResponse,
)

router = APIRouter(tags=["feedback"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)


def get_feedback_controller() -> FeedbackController:
    return FeedbackController()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    request: FeedbackRequest,
    controller: FeedbackController = Depends(get_feedback_controller),
) -> dict:
    logger.info("feedback endpoint received session_id=%s rating=%s", request.session_id, request.rating)
    try:
        return controller.submit(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("feedback endpoint failed session_id=%s", request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feedback submission failed. Check server logs for the request id.",
        )


@router.get("/feedback", response_model=FeedbackListResponse)
def list_feedback(
    session_id: str | None = None,
    limit: int | None = Query(default=None, ge=1, le=1000),
    controller: FeedbackController = Depends(get_feedback_controller),
) -> dict:
    logger.info("feedback list endpoint received session_id=%s limit=%s", session_id, limit)
    try:
        return controller.list_feedback(session_id=session_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("feedback list endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feedback lookup failed. Check server logs for the request id.",
        )


@router.get("/feedback/satisfaction", response_model=FeedbackSatisfactionResponse)
def get_satisfaction(
    controller: FeedbackController = Depends(get_feedback_controller),
) -> dict:
    logger.info("feedback satisfaction endpoint received")
    try:
        return controller.satisfaction_summary()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("feedback satisfaction endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feedback satisfaction lookup failed. Check server logs for the request id.",
        )
