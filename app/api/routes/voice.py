from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    try:
        # Process uploaded voice file
        return JSONResponse({"status": "success", "message": "Voice uploaded successfully"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})