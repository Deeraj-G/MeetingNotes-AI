"""
FastAPI app for the transcription service
"""

from uuid import UUID

from fastapi import FastAPI, File, UploadFile

from backend.transcribe.transcribe import transcribe_file

app = FastAPI()


@app.post("/{tenant_id}/transcribe")
async def transcribe(tenant_id: UUID, file: UploadFile = File(...)):
    """
    Transcribe a file and return the transcription
    """
    # Get the file content
    file_content = await file.read()

    # Transcribe the file
    transcription = await transcribe_file(tenant_id, file_content)

    return {"transcription": transcription}
