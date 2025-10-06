"""Text-to-Speech utilities backed by Google Gemini."""

from __future__ import annotations

import io
import logging
import os
import wave

from dotenv import load_dotenv

load_dotenv(".env")

# Configure logging
logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Configuration
USE_TTS = os.getenv("USE_TTS", "false").lower() == "true"
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")


def _check_dependencies() -> None:
    """Ensure required dependencies are available."""
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai is not available. Install it with `uv add google-genai`."
        )


def _get_client() -> genai.Client:
    """Return a configured Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("Set GEMINI_API_KEY in .env to use TTS functionality")

    return genai.Client(api_key=api_key)


def _generate_audio_data(text: str, voice: str = TTS_VOICE) -> bytes | None:
    """
    Convert text to speech using Google Gemini TTS and return audio data.

    Args:
        text: The text to convert to speech
        voice: The voice to use (default: Kore)

    Returns:
        WAV audio data as bytes, or None if TTS is disabled or failed

    Raises:
        RuntimeError: If dependencies are missing or API key is not configured
    """
    if not USE_TTS:
        return None

    if not text.strip():
        return None

    _check_dependencies()

    try:
        client = _get_client()

        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=f"Say cheerfully: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    )
                ),
            ),
        )

        pcm_data = response.candidates[0].content.parts[0].inline_data.data

        # Convert PCM to WAV format in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # 24kHz
            wf.writeframes(pcm_data)

        return wav_buffer.getvalue()

    except Exception as exc:
        logger.error(f"Failed to generate speech: {exc}")
        return None


def _play_audio_from_memory(audio_data: bytes) -> bool:
    """
    Play audio data directly from memory using sounddevice.

    Args:
        audio_data: WAV audio data as bytes

    Returns:
        True if playback was successful, False otherwise
    """
    try:
        import sounddevice as sd
        import soundfile as sf

        # Load audio data from memory
        audio_buffer = io.BytesIO(audio_data)
        audio_array, sample_rate = sf.read(audio_buffer, dtype="float32")

        # Play audio
        sd.play(audio_array, sample_rate)
        sd.wait()  # Wait until playback is finished

        return True

    except Exception as exc:
        logger.error(f"Failed to play audio from memory: {exc}")
        return False


def speak_and_play(text: str, voice: str = TTS_VOICE) -> bool:
    """
    Convert text to speech and play it immediately from memory.

    Args:
        text: The text to convert to speech
        voice: The voice to use (default: Kore)

    Returns:
        True if successful, False otherwise
    """
    audio_data = _generate_audio_data(text, voice)
    if audio_data:
        return _play_audio_from_memory(audio_data)
    return False


def speak_text(text: str, voice: str = TTS_VOICE) -> bytes | None:
    """
    Convert text to speech using Google Gemini TTS and return audio data.

    Args:
        text: The text to convert to speech
        voice: The voice to use (default: Kore)

    Returns:
        WAV audio data as bytes, or None if TTS is disabled or failed

    Raises:
        RuntimeError: If dependencies are missing or API key is not configured
    """
    return _generate_audio_data(text, voice)
