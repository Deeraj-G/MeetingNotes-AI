"""
Transcribe a file using OpenAI's Whisper API
"""

import os
import io

from pydub import AudioSegment

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


async def transcribe_file(tenant_id: str, file_content: bytes) -> str:
    """
    Transcribe a file and return the transcription
    """

    if not tenant_id or not file_content:
        missing = "Tenant ID" if not tenant_id else "File content"
        raise ValueError(f"{missing} is required")

    logger.info(f"file content type: {type(file_content)}")

    file = convert_to_file(file_content)

    # Upload the file to OpenAI
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        language="en",
        temperature=0.15,
        prompt="You are a helpful assistant that transcribes audio. You are given a file and you need to transcribe it.",
        response_format="text"
    )

    logger.info(f"Transcription response: {response.text}")

    return response.text

def convert_to_file(file_content: str):
    """
    Convert a bytes object to a file
    """

    # Convert file_content to MP3 format
    audio = AudioSegment.from_file(io.BytesIO(file_content))
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    return mp3_io.seek(0)

    
