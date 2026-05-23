import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.auth import verify_api_key
from app.core.dependencies import get_build_service
from app.controllers.build_controller import BuildController
from app.schemas.build import (
    FileBuildResponse,
    FileDeleteResponse,
    FileStatusResponse,
    UploadedFileResponse,
)
from app.services.rag.rag_builder import BuildService

router = APIRouter(tags=["build"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)


def get_build_controller(
    build_service: BuildService = Depends(get_build_service),
) -> BuildController:
    return BuildController(build_service)


@router.post(
    "/files",
    response_model=UploadedFileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile = File(...),
    controller: BuildController = Depends(get_build_controller),
) -> dict:
    logger.info("file upload endpoint received filename=%s", file.filename)
    try:
        return await controller.upload_file(file=file)
    except ValueError as exc:
        raise _upload_error(exc)
    except Exception:
        logger.exception("file upload endpoint failed filename=%s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed. Check server logs for the request id.",
        )


@router.post("/files/build", response_model=FileBuildResponse)
def build_files(
    file_id: str | None = None,
    controller: BuildController = Depends(get_build_controller),
) -> dict:
    logger.info("file build endpoint received file_id=%s", file_id)
    try:
        return controller.build_files(file_id=file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception:
        logger.exception("file build endpoint failed file_id=%s", file_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File build failed. Check server logs for the request id.",
        )


@router.get("/files", response_model=FileStatusResponse | list[FileStatusResponse])
def list_files(
    file_id: str | None = None,
    controller: BuildController = Depends(get_build_controller),
) -> dict | list[dict]:
    logger.info("file list endpoint received file_id=%s", file_id)
    try:
        if file_id:
            return controller.get_file(file_id)
        return controller.list_files()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception:
        logger.exception("file list endpoint failed file_id=%s", file_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File lookup failed. Check server logs for the request id.",
        )


@router.delete("/files", response_model=FileDeleteResponse)
def delete_files(
    file_id: str | None = None,
    controller: BuildController = Depends(get_build_controller),
) -> dict:
    logger.info("file delete endpoint received file_id=%s", file_id)
    try:
        return controller.delete_files(file_id=file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception:
        logger.exception("file delete endpoint failed file_id=%s", file_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File delete failed. Check server logs for the request id.",
        )

def _upload_error(exc: ValueError) -> HTTPException:
    detail = str(exc)
    if "too large" in detail:
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    elif "PDF" in detail:
        status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    else:
        status_code = status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=status_code, detail=detail)
