"""
Multi-engine Text-to-Speech module with factory pattern.

Supports ElevenLabs (best quality), Google TTS (free), and local Coqui TTS (free).
Uses factory pattern to switch engines via configuration.

Author: Gourav Pandey
"""

import io
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result from a TTS synthesis operation."""

    audio_bytes: bytes
    duration_s: float
    provider: str
    processing_time_s: float
    format: str  # 'mp3', 'wav'

    def save(self, path: str | Path) -> None:
        """Save audio to a file."""
        Path(path).write_bytes(self.audio_bytes)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Input text to convert to speech.

        Returns:
            TTSResult containing audio bytes and metadata.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this TTS engine is configured and available."""
        ...


class ElevenLabsTTS(TTSEngine):
    """
    ElevenLabs TTS engine - highest quality neural voice synthesis.

    Requires ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID environment variables.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = "eleven_monolingual_v1",
    ) -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "")
        self._model_id = model_id
        self._base_url = "https://api.elevenlabs.io/v1"

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using ElevenLabs API."""
        import requests

        start = time.time()

        url = f"{self._base_url}/text-to-speech/{self._voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self._api_key,
        }
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        audio_bytes = response.content
        processing_time = time.time() - start

        # Estimate duration (~150 words/min, ~5 chars/word)
        est_duration = len(text) / 5 / 150 * 60

        logger.info(
            "ElevenLabs TTS: %d chars -> %d bytes in %.2fs",
            len(text), len(audio_bytes), processing_time,
        )

        return TTSResult(
            audio_bytes=audio_bytes,
            duration_s=est_duration,
            provider="elevenlabs",
            processing_time_s=processing_time,
            format="mp3",
        )

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    def is_available(self) -> bool:
        return bool(self._api_key and self._voice_id)


class GoogleTTS(TTSEngine):
    """
    Google Text-to-Speech engine using gTTS library (free, no API key needed).
    """

    def __init__(self, language: str = "en", slow: bool = False) -> None:
        self._language = language
        self._slow = slow

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using Google TTS."""
        from gtts import gTTS

        start = time.time()

        tts = gTTS(text=text, lang=self._language, slow=self._slow)

        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        audio_bytes = buffer.getvalue()

        processing_time = time.time() - start
        est_duration = len(text) / 5 / 150 * 60

        logger.info(
            "Google TTS: %d chars -> %d bytes in %.2fs",
            len(text), len(audio_bytes), processing_time,
        )

        return TTSResult(
            audio_bytes=audio_bytes,
            duration_s=est_duration,
            provider="google",
            processing_time_s=processing_time,
            format="mp3",
        )

    @property
    def provider_name(self) -> str:
        return "google"

    def is_available(self) -> bool:
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False


class CoquiTTS(TTSEngine):
    """
    Local Coqui TTS engine - free, runs entirely offline.

    Requires TTS package: pip install TTS
    """

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> None:
        self._model_name = model_name
        self._tts = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the Coqui TTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS
                self._tts = TTS(model_name=self._model_name, progress_bar=False)
                logger.info("Coqui TTS model loaded: %s", self._model_name)
            except ImportError:
                raise RuntimeError("Coqui TTS not installed. Run: pip install TTS")

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech using local Coqui TTS."""
        self._ensure_loaded()

        start = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            self._tts.tts_to_file(text=text, file_path=tmp.name)
            audio_bytes = Path(tmp.name).read_bytes()

        processing_time = time.time() - start
        est_duration = len(text) / 5 / 150 * 60

        logger.info(
            "Coqui TTS: %d chars -> %d bytes in %.2fs",
            len(text), len(audio_bytes), processing_time,
        )

        return TTSResult(
            audio_bytes=audio_bytes,
            duration_s=est_duration,
            provider="coqui",
            processing_time_s=processing_time,
            format="wav",
        )

    @property
    def provider_name(self) -> str:
        return "coqui"

    def is_available(self) -> bool:
        try:
            from TTS.api import TTS
            return True
        except ImportError:
            return False


class TTSFactory:
    """
    Factory for creating TTS engines based on configuration.

    Supports automatic fallback: if the preferred engine is unavailable,
    falls back to the next available engine in priority order.
    """

    _engines: dict[str, type[TTSEngine]] = {
        "elevenlabs": ElevenLabsTTS,
        "google": GoogleTTS,
        "coqui": CoquiTTS,
    }

    _priority_order = ["elevenlabs", "google", "coqui"]

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        **kwargs: Any,
    ) -> TTSEngine:
        """
        Create a TTS engine instance.

        Args:
            provider: Provider name ('elevenlabs', 'google', 'coqui').
                If None, uses TTS_PROVIDER env var or auto-selects.
            **kwargs: Additional arguments passed to the engine constructor.

        Returns:
            Configured TTSEngine instance.

        Raises:
            ValueError: If no TTS engine is available.
        """
        provider = provider or os.environ.get("TTS_PROVIDER", "")

        if provider and provider in cls._engines:
            engine = cls._engines[provider](**kwargs)
            if engine.is_available():
                logger.info("TTS engine selected: %s", provider)
                return engine
            logger.warning("Requested TTS provider '%s' is not available", provider)

        # Fallback: try engines in priority order
        for name in cls._priority_order:
            try:
                engine = cls._engines[name]()
                if engine.is_available():
                    logger.info("TTS engine fallback to: %s", name)
                    return engine
            except Exception as e:
                logger.debug("TTS engine '%s' not available: %s", name, e)

        raise ValueError("No TTS engine available. Install gTTS for basic support: pip install gTTS")

    @classmethod
    def list_available(cls) -> list[str]:
        """Return names of all available TTS engines."""
        available = []
        for name in cls._priority_order:
            try:
                engine = cls._engines[name]()
                if engine.is_available():
                    available.append(name)
            except Exception:
                pass
        return available
