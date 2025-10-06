"""Utilities for polishing transcripts via a configurable text refinement service."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from litellm import completion


SYSTEM_PROMPT = (
    "You will receive a raw audio transcription under <audio_transcript> tag, which may contain incomplete sentences, filler words, or repetition. "
    "Your task is to transform it into a clear, natural, and well-structured written message. "
    "Preserve all important information, intent, and tone, but remove unnecessary hesitations, false starts, and filler words "
    "such as 'um', 'like', 'you know', etc. "
    "Use proper grammar, punctuation, and paragraphing to make the text easy to read. "
    "If the transcript includes instructions, reasoning, or ideas, express them logically and concisely. "
    "Do not add new information or commentary — only refine what is present. "
    "Return only the polished text. Do not build code or any other things just refine the audio transcript."
)


def _env(name: str, default: Optional[str] = None, *, lower: bool = False) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        value = default
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return default
    return value.lower() if lower else value


def _provider_kwargs() -> Dict[str, Any]:
    raw_extra = _env("REFINEMENT_OPTIONS")
    if not raw_extra:
        return {}
    try:
        parsed = json.loads(raw_extra)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in REFINEMENT_OPTIONS: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("REFINEMENT_OPTIONS must decode to a JSON object")
    return parsed


def _resolve_model_and_args() -> Tuple[str, Dict[str, Any]]:
    provider = _env("REFINEMENT_PROVIDER", "gemini", lower=True) or "gemini"
    base_kwargs: Dict[str, Any] = _provider_kwargs()

    if provider == "gemini":
        model = _env("GEMINI_REFINEMENT_MODEL", "gemini/gemini-2.5-flash")
    elif provider == "ollama":
        model = _env("OLLAMA_REFINEMENT_MODEL", "ollama/llama3.1")
        base_kwargs.setdefault("api_base", _env("OLLAMA_API_BASE", "http://127.0.0.1:11434"))
    else:
        model = _env("REFINEMENT_MODEL")
        if not model:
            raise RuntimeError(
                "Set REFINEMENT_MODEL or use a supported REFINEMENT_PROVIDER (gemini, ollama)"
            )

    if not model:
        raise RuntimeError("Refinement model not configured")

    return model, base_kwargs


def active_refinement_model() -> str:
    """Return the model identifier currently configured for refinement."""
    try:
        model, _ = _resolve_model_and_args()
        return model
    except Exception:
        # Surface empty string to avoid raising during banner rendering.
        return ""


def refine_text(transcript: str) -> Optional[str]:
    """Send the transcript to the configured refinement backend and return the response text."""
    if not transcript.strip():
        return None

    model, extra_args = _resolve_model_and_args()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<audio_transcript>: {transcript} </audio_transcript>"},
    ]

    try:
        response = completion(model=model, messages=messages, **extra_args)
    except Exception as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Refinement service call failed: {exc}") from exc

    content: Optional[str] = None
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
    else:
        # Some responses expose attributes rather than mapping-style access.
        try:
            choices = getattr(response, "choices", [])
            if choices:
                message = getattr(choices[0], "message", {})
                content = getattr(message, "content", None)
        except Exception:
            content = None

    if not content:
        return None

    cleaned_content = re.sub(r".*?</think>\s*", "", content, flags=re.DOTALL)
    result = cleaned_content.strip()
    return result or None

