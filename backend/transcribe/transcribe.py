"""
Transcribe a file using OpenAI's Whisper API
"""

import os
from uuid import UUID

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


async def transcribe_file(tenant_id: UUID, file_content: bytes) -> str:
    """
    Transcribe a file and return the transcription
    """

    if not tenant_id or not file_content:
        missing = "Tenant ID" if not tenant_id else "File content"
        raise ValueError(f"{missing} is required")

    # Upload the file to OpenAI
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_content,
        language="en",
        temperature=0.15,
        prompt="You are a helpful assistant that transcribes audio. You are given a file and you need to transcribe it.",
        response_format="text"
    )

    logger.info(f"Transcription response: {response.text}")

    return response.text
