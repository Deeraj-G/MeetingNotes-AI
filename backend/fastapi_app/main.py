"""
FastAPI app for the transcription service
"""

from base64 import b64decode

from fastapi import FastAPI, Form, status
from loguru import logger

from backend.transcribe.transcribe import transcribe_file

app = FastAPI()


@app.post("/{tenant_id}/transcribe")
async def transcribe(tenant_id: str, data: str = Form(...)):
    """
    Transcribe a file and return the transcription
    """

    logger.info(f"tenant_id: {tenant_id}")
    try:
        # Get the file content
        file_content = b64decode(data.split(",")[1])
    except (TimeoutError, ConnectionError) as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = str(e)
        return {"error": detail, "status_code": status_code}

    try:
        # Transcribe the file
        transcription = await transcribe_file(tenant_id, file_content)
    except ValueError as e:
        status_code = status.HTTP_400_BAD_REQUEST
        detail = str(e)
        return {"error": detail, "status_code": status_code}

    return {"transcription": transcription}
