from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    id: str
    message: str

