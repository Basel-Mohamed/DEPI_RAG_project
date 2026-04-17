@router.post("/build")
async def build(file: UploadFile):

    await rag_build_pipeline.run(file.filename)

    return {"status": "index built"}