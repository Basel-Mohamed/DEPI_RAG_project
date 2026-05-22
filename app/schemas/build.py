from enum import Enum

from pydantic import BaseModel, Field


class FileBuildStatus(str, Enum):
    UPLOADED = "uploaded"
    BUILDING = "building"
    BUILT = "built"
    FAILED = "failed"
    MISSING = "missing"


class UploadedFileResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str | None = None
    status: FileBuildStatus


class FileStatusResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str | None = None
    status: FileBuildStatus
    chunks_count: int = Field(ge=0)
    page_images_count: int = Field(default=0, ge=0)
    chunks_with_page_images_count: int = Field(default=0, ge=0)
    upserted: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    last_error: str | None = None


class FileBuildResponse(BaseModel):
    files: list[FileStatusResponse]


class FileDeleteResponse(BaseModel):
    file_id: str | None = None
    deleted_count: int = Field(ge=0)
    files_deleted: int = Field(ge=0)
