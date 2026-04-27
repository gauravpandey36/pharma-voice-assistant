"""
Full voice pipeline: audio_in -> ASR -> RAG -> LLM -> TTS -> audio_out.

Orchestrates all components with latency budgeting and logging at each stage.

Author: Gourav Pandey
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .asr import WhisperASR, TranscriptionResult
from .graceful_degradation import DegradationHandler
from .llm_engine import LLMEngine, LLMResponse
from .tts import TTSFactory, TTSResult

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Latency measurements for each pipeline stage."""

    asr_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to serializable dictionary."""
        return {
            "asr_ms": round(self.asr_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "llm_ttft_ms": round(self.llm_ttft_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass
class PipelineResult:
    """Complete result from the voice pipeline."""

    query_text: str
    answer_text: str
    audio_bytes: bytes | None
    audio_format: str
    latency: LatencyBreakdown
    sources: list[str] = field(default_factory=list)
    model_used: str = ""
    tts_provider: str = ""
    degraded: bool = False
    degradation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "query_text": self.query_text,
            "answer_text": self.answer_text,
            "has_audio": self.audio_bytes is not None,
            "audio_format": self.audio_format,
            "latency": self.latency.to_dict(),
            "sources": self.sources,
            "model_used": self.model_used,
            "tts_provider": self.tts_provider,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
        }


class VoicePipeline:
    """
    End-to-end voice assistant pipeline.

    Processes voice or text queries through ASR, RAG retrieval, LLM generation,
    and TTS synthesis with latency budgeting and graceful degradation.
    """

    def __init__(
        self,
        asr: WhisperASR | None = None,
        llm_engine: LLMEngine | None = None,
        tts_provider: str | None = None,
        retriever: Any = None,
    ) -> None:
        """
        Initialize the voice pipeline.

        Args:
            asr: Whisper ASR instance (created lazily if None).
            llm_engine: LLM engine instance (created lazily if None).
            tts_provider: Preferred TTS provider name.
            retriever: Optional RAG retriever for context augmentation.
        """
        self._asr = asr
        self._llm = llm_engine or LLMEngine()
        self._tts_provider = tts_provider
        self._retriever = retriever
        self._degradation = DegradationHandler()

        logger.info("Voice pipeline initialized")

    def _get_asr(self) -> WhisperASR:
        """Get or create the ASR engine."""
        if self._asr is None:
            self._asr = WhisperASR()
        return self._asr

    def _get_tts(self):
        """Get or create the TTS engine."""
        return TTSFactory.create(provider=self._tts_provider)

    def _retrieve_context(self, query: str) -> tuple[str, list[str], float]:
        """
        Retrieve context from the knowledge base.

        Args:
            query: Search query.

        Returns:
            Tuple of (context_text, source_ids, latency_ms).
        """
        if self._retriever is None:
            return "", [], 0.0

        start = time.time()
        try:
            results = self._retriever.search(query, top_k=5)
            context = "\n\n".join(
                r.get("text", "") if isinstance(r, dict) else getattr(r, "text", "")
                for r in results
            )
            sources = [
                r.get("chunk_id", "") if isinstance(r, dict) else getattr(r, "chunk_id", "")
                for r in results
            ]
            latency = (time.time() - start) * 1000
            return context, sources, latency
        except Exception as e:
            logger.warning("Retrieval failed: %s", e)
            latency = (time.time() - start) * 1000
            return "", [], latency

    def process_voice(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
    ) -> PipelineResult:
        """
        Process a voice query end-to-end.

        Pipeline: audio -> ASR -> RAG -> LLM -> TTS -> audio

        Args:
            audio_bytes: Raw audio bytes.
            audio_format: Audio format ('wav', 'mp3', 'webm').

        Returns:
            PipelineResult with answer text and audio.
        """
        total_start = time.time()
        latency = LatencyBreakdown()

        # Stage 1: ASR
        asr_start = time.time()
        try:
            asr = self._get_asr()
            transcription = asr.transcribe_bytes(audio_bytes, file_format=audio_format)
            query_text = transcription.text
            latency.asr_ms = (time.time() - asr_start) * 1000
            logger.info("ASR: '%s' (%.0fms)", query_text[:50], latency.asr_ms)
        except Exception as e:
            latency.asr_ms = (time.time() - asr_start) * 1000
            return self._degradation.handle_asr_failure(e, latency, total_start)

        return self._process_text_internal(query_text, latency, total_start)

    def process_text(self, query: str) -> PipelineResult:
        """
        Process a text query (skipping ASR).

        Pipeline: text -> RAG -> LLM -> TTS -> audio

        Args:
            query: Text query.

        Returns:
            PipelineResult with answer text and audio.
        """
        total_start = time.time()
        latency = LatencyBreakdown()
        return self._process_text_internal(query, latency, total_start)

    def _process_text_internal(
        self,
        query: str,
        latency: LatencyBreakdown,
        total_start: float,
    ) -> PipelineResult:
        """Internal text processing pipeline."""

        # Stage 2: RAG Retrieval
        context, sources, retrieval_ms = self._retrieve_context(query)
        latency.retrieval_ms = retrieval_ms

        # Stage 3: LLM Generation
        llm_start = time.time()
        try:
            llm_response = self._llm.generate(query, context=context)
            answer_text = llm_response.text
            latency.llm_ms = (time.time() - llm_start) * 1000
            latency.llm_ttft_ms = llm_response.ttft_s * 1000
            model_used = f"{llm_response.provider}/{llm_response.model}"
            logger.info("LLM: %d chars (%.0fms, TTFT %.0fms)", len(answer_text), latency.llm_ms, latency.llm_ttft_ms)
        except Exception as e:
            latency.llm_ms = (time.time() - llm_start) * 1000
            return self._degradation.handle_llm_failure(e, query, latency, total_start)

        # Stage 4: TTS
        tts_start = time.time()
        try:
            tts = self._get_tts()
            tts_result = tts.synthesize(answer_text)
            audio_bytes = tts_result.audio_bytes
            audio_format = tts_result.format
            tts_provider = tts_result.provider
            latency.tts_ms = (time.time() - tts_start) * 1000
            logger.info("TTS: %d bytes (%.0fms)", len(audio_bytes), latency.tts_ms)
        except Exception as e:
            latency.tts_ms = (time.time() - tts_start) * 1000
            return self._degradation.handle_tts_failure(
                e, query, answer_text, sources, model_used, latency, total_start
            )

        latency.total_ms = (time.time() - total_start) * 1000

        return PipelineResult(
            query_text=query,
            answer_text=answer_text,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            latency=latency,
            sources=sources,
            model_used=model_used,
            tts_provider=tts_provider,
        )
