import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.controllers.inference_controller import InferenceController
from app.controllers.monitoring_controller import MonitoringMetrics
from app.core.auth import verify_api_key
from app.core.dependencies import get_inference_pipeline
from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services.rag.rag_inference import RagInferencePipeline

router = APIRouter(tags=["inference"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)


def get_inference_controller(
    pipeline: RagInferencePipeline = Depends(get_inference_pipeline),
) -> InferenceController:
    return InferenceController(pipeline)


@router.post("/ask", response_model=InferenceResponse)
def ask_question(
    request: InferenceRequest,
    controller: InferenceController = Depends(get_inference_controller),
) -> dict:
    logger.info("ask endpoint received")
    try:
        return controller.ask(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("ask endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Check server logs for the request id.",
        )


@router.post("/ask/stream")
def stream_answer(
    http_request: Request,
    request: InferenceRequest,
    controller: InferenceController = Depends(get_inference_controller),
) -> StreamingResponse:
    http_request.state.skip_monitoring_metrics = True
    logger.info("ask stream endpoint received")
    try:
        stream = controller.stream(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    start = time.perf_counter()

    def measured_stream():
        try:
            yield from stream
        finally:
            MonitoringMetrics.record_request_latency(
                (time.perf_counter() - start) * 1000
            )

    return StreamingResponse(measured_stream(), media_type="application/x-ndjson")
