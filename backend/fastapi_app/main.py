"""
FastAPI app for the transcription service
"""

from uuid import UUID

from fastapi import FastAPI, File, UploadFile, status

from backend.transcribe.transcribe import transcribe_file

app = FastAPI()


@app.post("/{tenant_id}/transcribe")
async def transcribe(tenant_id: UUID, file: UploadFile = File(...)):
    """
    Transcribe a file and return the transcription
    """
    try:
        # Get the file content
        file_content = await file.read()
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
