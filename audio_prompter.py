"""Audio Prompter: hold Ctrl+Shift to record speech and release the keys to finish."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(".env")
import io
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf
from pynput import keyboard

try:  # Windows notification sound
    import winsound
except ImportError:  # pragma: no cover - non-Windows
    winsound = None  # type: ignore[assignment]

from refinement import active_refinement_model, refine_text
from text_to_speech import speak_and_play

# Optional import (loaded lazily when Whisper backend is selected)
try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - only used when backend is whisper
    WhisperModel = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SAMPLE_RATE_TARGET = 16_000
CHUNK_SECONDS = 15
CHANNELS = 1


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BACKEND = os.getenv("TRANSCRIPTION_BACKEND", "gemini").lower()
USE_REFINEMENT = _env_bool("USE_REFINEMENT", True)
GEMINI_STT_MODEL = os.getenv("GEMINI_STT_MODEL", "models/gemini-2.5-flash")
USE_TTS = _env_bool("USE_TTS", False)


def _deliver_text(text: str, *, refined: bool) -> None:
    label = "Refined transcript" if refined else "Full transcript"
    try:
        pyperclip.copy(text)
        if winsound:
            winsound.MessageBeep()
        logger.info(f"{label} copied to clipboard")
    except pyperclip.PyperclipException as exc:
        logger.error(f"Failed to copy to clipboard: {exc}")
        return

    if USE_TTS:
        try:
            speak_and_play(text)
            logger.info("Text spoken aloud")
        except Exception as exc:
            logger.error(f"Failed to speak text: {exc}")


def _maybe_refine_text(full_text: str) -> tuple[str, bool]:
    if not USE_REFINEMENT:
        return full_text, False

    refined_text: str | None = None
    try:
        refined_text = refine_text(full_text)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Refinement failed: {exc}")

    if refined_text:
        logger.info(f"Refined text: {refined_text}")
        return refined_text, True

    return full_text, False


@dataclass
class AppState:
    recording: bool = False
    session_id: int | None = None
    next_session: int = 1
    buffer: list[np.ndarray] = field(default_factory=list)
    transcripts: dict[int, list[str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start_session(self) -> int:
        with self.lock:
            self.recording = True
            self.buffer.clear()
            self.session_id = self.next_session
            self.transcripts[self.session_id] = []
            self.next_session += 1
            return self.session_id

    def stop_session(self) -> int | None:
        with self.lock:
            self.recording = False
            return self.session_id

    def clear_session(self) -> None:
        with self.lock:
            self.session_id = None

    def add_buffer_chunk(self, chunk: np.ndarray) -> None:
        with self.lock:
            if self.recording:
                self.buffer.append(chunk)

    def pop_buffer(self, samples: int | None, samplerate: int) -> np.ndarray:
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            data = np.concatenate(self.buffer).reshape(-1)
            self.buffer.clear()
        if samples is None:
            return data
        return data[: min(samples, data.shape[0])]

    def append_transcript(self, session_id: int, text: str) -> None:
        with self.lock:
            self.transcripts.setdefault(session_id, []).append(text)

    def finish_transcript(self, session_id: int) -> str:
        with self.lock:
            parts = self.transcripts.pop(session_id, [])
        return " ".join(filter(None, parts)).strip()


STATE = AppState()
AUDIO_QUEUE: queue.Queue[tuple[np.ndarray, int, int, bool]] = queue.Queue()
STOP_EVENT = threading.Event()


def _resample(audio: np.ndarray, original_sr: int) -> np.ndarray:
    if original_sr == SAMPLE_RATE_TARGET:
        return audio.astype(np.float32).reshape(-1)
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    duration = audio.shape[0] / float(original_sr)
    target_samples = int(round(duration * SAMPLE_RATE_TARGET))
    if target_samples <= 1:
        return audio.astype(np.float32).reshape(-1)
    src = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=True)
    dst = np.linspace(0.0, 1.0, num=target_samples, endpoint=True)
    return np.interp(dst, src, audio.astype(np.float32).reshape(-1)).astype(np.float32)


def _build_transcriber() -> Callable[[np.ndarray, int], str]:
    if BACKEND == "whisper":
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not available")
        logger.info("Loading Whisper model (base)")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper ready")

        def transcribe(audio: np.ndarray, samplerate: int) -> str:
            audio_16k = _resample(audio, samplerate)
            if audio_16k.size == 0:
                return ""
            segments, _ = whisper_model.transcribe(audio_16k, language=None)
            return " ".join(segment.text for segment in segments).strip()

        return transcribe

    if BACKEND == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise RuntimeError("Set GEMINI_API_KEY in .env to use the Gemini backend")
        import google.generativeai as genai
        from google.api_core import retry as api_retry

        genai.configure(api_key=api_key)
        logger.info("Gemini API configured")
        model = genai.GenerativeModel(
            GEMINI_STT_MODEL,
            system_instruction="You are a speech-to-text transcriber. Your only job is to transcribe spoken words from audio. Return ONLY the transcribed text with no explanations, commentary, or additional formatting.",
        )

        retry_strategy = api_retry.Retry(
            initial=0.5,
            multiplier=2,
            maximum=60,
            timeout=300,
            predicate=api_retry.if_exception_type(Exception),
        )

        def transcribe(audio: np.ndarray, samplerate: int) -> str:
            if audio.size == 0:
                return ""
            wav_buffer = io.BytesIO()
            with sf.SoundFile(
                wav_buffer,
                mode="w",
                samplerate=samplerate,
                channels=1,
                format="WAV",
                subtype="PCM_16",
            ) as wav_file:
                wav_file.write((audio * 32767).astype(np.int16))
            blob = {
                "inline_data": {"mime_type": "audio/wav", "data": wav_buffer.getvalue()}
            }
            response = model.generate_content(
                [
                    "Transcribe the speech in this audio file. Return ONLY the spoken words, nothing else. Do not include explanations, analysis, or any other text.",
                    blob,
                ],
                request_options={"retry": retry_strategy},
            )
            return (response.text or "").strip()

        return transcribe

    raise ValueError(f"Unsupported backend: {BACKEND}")


TRANSCRIBE = _build_transcriber()


def audio_callback(
    indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
) -> None:
    if status:
        logger.warning(f"Audio warning: {status}")
    STATE.add_buffer_chunk(indata.copy())


def recording_loop() -> None:
    stream: sd.InputStream | None = None
    samplerate = SAMPLE_RATE_TARGET
    target_samples = int(CHUNK_SECONDS * samplerate)

    while not STOP_EVENT.is_set():
        if STATE.recording and stream is None:
            samplerate = _pick_input_samplerate()
            target_samples = int(CHUNK_SECONDS * samplerate)
            stream = sd.InputStream(
                samplerate=samplerate,
                channels=CHANNELS,
                dtype="float32",
                callback=audio_callback,
            )
            stream.start()
            logger.info(f"Recording at {samplerate} Hz")

        if not STATE.recording and stream is not None:
            session_id = STATE.session_id
            data = STATE.pop_buffer(None, samplerate)
            if session_id is not None:
                AUDIO_QUEUE.put((data, samplerate, session_id, True))
            stream.stop()
            stream.close()
            stream = None
            STATE.clear_session()
            logger.info("Recording stopped")

        if STATE.recording and stream is not None and STATE.session_id is not None:
            total_samples = sum(len(chunk) for chunk in STATE.buffer)
            if total_samples >= target_samples:
                data = STATE.pop_buffer(target_samples, samplerate)
                AUDIO_QUEUE.put((data, samplerate, STATE.session_id, False))
        time.sleep(0.05)

    if stream is not None:  # Cleanup
        stream.stop()
        stream.close()


def transcription_loop() -> None:
    while True:
        if STOP_EVENT.is_set() and AUDIO_QUEUE.empty():
            break
        try:
            audio, samplerate, session_id, is_final = AUDIO_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            text = TRANSCRIBE(audio, samplerate)
        except Exception as exc:  # pragma: no cover
            text = f"[Transcription Error] {exc}"

        if text:
            logger.info(f"Live transcription: {text}")
        elif not is_final:
            logger.debug("Live transcription: (silence)")

        STATE.append_transcript(session_id, text)

        if is_final:
            full_text = STATE.finish_transcript(session_id)
            if full_text:
                logger.info(f"Full transcription: {full_text}")
                final_text, refined = _maybe_refine_text(full_text)
                _deliver_text(final_text, refined=refined)
            else:
                logger.debug("Full transcription: (silence)")

        AUDIO_QUEUE.task_done()


class HotkeyListener:
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.ctrl = False
        self.shift = False

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl = True
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift = True
        if self.ctrl and self.shift and not STATE.recording:
            session = STATE.start_session()
            logger.info(f"Session {session} started (hold Ctrl+Shift)")

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl = False
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift = False
        if STATE.recording and not (self.ctrl and self.shift):
            STATE.stop_session()
            logger.info("Finishing transcription")
        if key == keyboard.Key.esc:
            STATE.stop_session()
            self.stop_event.set()
            logger.info("Stopping application")
            return False
        return None


def _pick_input_samplerate() -> int:
    try:
        default_input = sd.default.device[0]
        info = sd.query_devices(default_input)
        return int(info.get("default_samplerate", SAMPLE_RATE_TARGET))
    except Exception:  # pragma: no cover
        return SAMPLE_RATE_TARGET


def _print_banner() -> None:
    logger.info("=" * 60)
    logger.info("Audio Prompter - Real-time Speech Transcription")
    logger.info("=" * 60)
    logger.info(f"Backend            : {BACKEND.upper()}")
    logger.info(f"Refinement         : {'ON' if USE_REFINEMENT else 'OFF'}")
    if USE_REFINEMENT:
        model_name = active_refinement_model()
        if model_name:
            logger.info(f"Refinement model   : {model_name}")
    logger.info(f"Text-to-Speech     : {'ON' if USE_TTS else 'OFF'}")
    logger.info(f"Chunk duration     : {CHUNK_SECONDS} s")
    logger.info(f"Target sample rate : {SAMPLE_RATE_TARGET} Hz")
    logger.info("")
    logger.info("Press & hold Ctrl+Shift to record, release to stop, Esc to quit.")
    logger.info("(Tip: Launch the app, then use the hotkey from anywhere.)")
    logger.info("=" * 60)
    logger.info("")


def main() -> None:
    _print_banner()

    recorder = threading.Thread(target=recording_loop, daemon=True)
    transcriber = threading.Thread(target=transcription_loop, daemon=True)
    recorder.start()
    transcriber.start()

    listener = HotkeyListener(STOP_EVENT)
    keyboard_listener = keyboard.Listener(
        on_press=listener.on_press,
        on_release=listener.on_release,
        suppress=False,
    )
    keyboard_listener.start()

    try:
        keyboard_listener.join()
    except KeyboardInterrupt:
        STOP_EVENT.set()
    finally:
        STOP_EVENT.set()
        keyboard_listener.stop()
        AUDIO_QUEUE.join()
        logger.info("Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
