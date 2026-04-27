"""
Whisper ASR (Automatic Speech Recognition) module.

Accepts audio files or streams and returns transcriptions with timestamps
and confidence scores. Supports multiple Whisper model sizes for
quality/speed tradeoffs.

Author: Gourav Pandey
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import whisper

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ASR_MODEL", "base")


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed speech."""

    text: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""

    text: str
    language: str
    segments: list[TranscriptionSegment]
    duration_s: float
    processing_time_s: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "segments": [
                {
                    "text": s.text,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "confidence": round(s.confidence, 3),
                }
                for s in self.segments
            ],
            "duration_s": round(self.duration_s, 2),
            "processing_time_s": round(self.processing_time_s, 3),
            "model_name": self.model_name,
        }


class WhisperASR:
    """
    Whisper-based ASR engine for pharmaceutical voice input.

    Supports transcription from audio files and raw audio bytes.
    Automatically detects language and provides word-level timestamps.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize the Whisper ASR engine.

        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                Larger models are more accurate but slower.
        """
        self._model_name = model_name
        self._model: whisper.Whisper | None = None
        logger.info("WhisperASR initialized with model: %s", model_name)

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            logger.info("Loading Whisper model '%s'...", self._model_name)
            start = time.time()
            self._model = whisper.load_model(self._model_name)
            elapsed = time.time() - start
            logger.info("Whisper model loaded in %.2fs", elapsed)

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).
            language: Optional language code (e.g., 'en'). Auto-detected if None.

        Returns:
            TranscriptionResult with full text and segments.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_model_loaded()

        logger.info("Transcribing file: %s", audio_path.name)
        start = time.time()

        options = {"fp16": False}
        if language:
            options["language"] = language

        result = self._model.transcribe(str(audio_path), **options)
        processing_time = time.time() - start

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptionSegment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    confidence=seg.get("avg_logprob", 0.0),
                )
            )

        # Estimate audio duration from last segment
        duration = segments[-1].end if segments else 0.0

        transcription = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            segments=segments,
            duration_s=duration,
            processing_time_s=processing_time,
            model_name=self._model_name,
        )

        logger.info(
            "Transcription complete: %d chars, %d segments, %.2fs processing",
            len(transcription.text), len(segments), processing_time,
        )
        return transcription

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        file_format: str = "wav",
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio from raw bytes.

        Writes bytes to a temporary file and delegates to transcribe_file.

        Args:
            audio_bytes: Raw audio bytes.
            file_format: Audio format extension ('wav', 'mp3', 'webm').
            language: Optional language code.

        Returns:
            TranscriptionResult.
        """
        with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.transcribe_file(tmp.name, language=language)

    def transcribe_numpy(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio from a numpy array.

        Args:
            audio_array: Audio samples as numpy array (float32, mono).
            sample_rate: Sample rate in Hz.
            language: Optional language code.

        Returns:
            TranscriptionResult.
        """
        self._ensure_model_loaded()

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            duration = len(audio_array) / sample_rate
            new_length = int(duration * 16000)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), new_length),
                np.arange(len(audio_array)),
                audio_array,
            ).astype(np.float32)

        start = time.time()

        options = {"fp16": False}
        if language:
            options["language"] = language

        result = self._model.transcribe(audio_array, **options)
        processing_time = time.time() - start

        segments = [
            TranscriptionSegment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                confidence=seg.get("avg_logprob", 0.0),
            )
            for seg in result.get("segments", [])
        ]

        duration = segments[-1].end if segments else len(audio_array) / 16000

        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            segments=segments,
            duration_s=duration,
            processing_time_s=processing_time,
            model_name=self._model_name,
        )

    @property
    def model_name(self) -> str:
        """Return the name of the loaded Whisper model."""
        return self._model_name
