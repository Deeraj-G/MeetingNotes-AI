"""
Transcribe a file using OpenAI's Whisper API
"""

import io
import os

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydub import AudioSegment

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

    try:
        file = convert_to_file(file_content)
    except Exception as e:
        logger.error(f"Error converting file: {e}")
        raise e

    logger.info(f"FILE: {file}")

    # Upload the file to OpenAI
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.mp3", file),
        language="en",
        temperature=0.15,
        prompt=(
            "Transcribe the conversation, performing speaker diarization. "
            "Identify each speaker and label their utterances consistently (e.g., 'speaker_1', 'speaker_2', ...). "
            "The output must be a Python list of strings, ordered chronologically. "
            "Each string in the list must strictly follow the format: 'speaker_N: Transcribed text'. "
            "Example of expected output structure: "
            "['speaker_1: First utterance.', 'speaker_2: Response utterance.', 'speaker_1: Follow-up utterance.']"
        ),
        response_format="verbose_json",
    )

    logger.info(f"Transcription response: {type(response)}")
    logger.info(f"Transcription response: {response}")

    return response


def convert_to_file(file_content: bytes) -> io.BytesIO:
    """
    Convert input audio bytes to an MP3 in-memory file-like object.
    """
    try:
        # Load audio from bytes. Pydub attempts to detect the format
        audio_segment = AudioSegment.from_file(io.BytesIO(file_content))

        # Create an in-memory bytes buffer
        output_io = io.BytesIO()

        # Export the audio segment to the buffer in mp3 format
        audio_segment.export(output_io, format="mp3")

        # Reset the buffer's position to the beginning
        output_io.seek(0)
        return output_io
    except FileNotFoundError as e:
        logger.error("Error: Audio file not found.")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
