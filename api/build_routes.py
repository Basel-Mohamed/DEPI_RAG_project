from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from pipelines.rag_build_pipeline import build_default_rag_build_pipeline


router = APIRouter(tags=["build"])

UPLOAD_ROOT = Path("data/uploads")


def _safe_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    return candidate or "uploaded-document"


@router.post("/build")
async def build(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required.")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_ROOT / _safe_filename(file.filename)

    async with aiofiles.open(target_path, "wb") as output_file:
        while chunk := await file.read(1024 * 1024):
            await output_file.write(chunk)

    try:
        pipeline = build_default_rag_build_pipeline()
    except (ImportError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    build_result = await pipeline.run(str(target_path))

    return {
        "status": "index built",
        "file_path": str(target_path),
        **build_result,
    }
