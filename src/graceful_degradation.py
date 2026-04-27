"""
Graceful degradation handler for the voice pipeline.

Manages timeout handling and fallback logic:
- ASR failure -> prompt for text input
- TTS failure -> return text only
- LLM timeout -> return cached/fallback response
All with user-friendly messages.

Author: Gourav Pandey
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Fallback responses for common pharmaceutical queries
FALLBACK_RESPONSES: dict[str, str] = {
    "default": (
        "I apologize, but I am experiencing technical difficulties processing "
        "your request at the moment. Please try again shortly, or rephrase "
        "your question. For urgent compliance questions, please refer to the "
        "relevant regulatory guidance documents directly."
    ),
    "capa": (
        "CAPA (Corrective and Preventive Action) is a systematic approach required "
        "by GMP regulations. It includes root cause analysis, corrective actions "
        "to address the immediate issue, preventive actions to prevent recurrence, "
        "and effectiveness verification. For detailed guidance, refer to ICH Q10 "
        "Section 3.2 and 21 CFR 211.192."
    ),
    "deviation": (
        "A deviation is any departure from an approved procedure, specification, "
        "or standard. Deviations must be documented, investigated for root cause, "
        "assessed for impact on product quality, and resolved through CAPA. "
        "Critical and major deviations require QA approval before batch release. "
        "Refer to your site deviation SOP and ICH Q10 for guidance."
    ),
    "validation": (
        "Process validation per FDA guidance follows a lifecycle approach: "
        "Stage 1 (Process Design), Stage 2 (Process Qualification with PPQ), "
        "and Stage 3 (Continued Process Verification). A minimum of three "
        "consecutive successful batches is expected for PPQ. Refer to FDA "
        "Process Validation Guidance (2011) and EU GMP Annex 15."
    ),
}


def _find_fallback(query: str) -> str:
    """
    Find the most relevant fallback response for a query.

    Args:
        query: User query text.

    Returns:
        Best matching fallback response text.
    """
    query_lower = query.lower()

    keyword_map = {
        "capa": ["capa", "corrective", "preventive"],
        "deviation": ["deviation", "non-conformance", "ncr"],
        "validation": ["validation", "qualify", "ppq", "iq", "oq", "pq"],
    }

    for key, keywords in keyword_map.items():
        if any(kw in query_lower for kw in keywords):
            return FALLBACK_RESPONSES[key]

    return FALLBACK_RESPONSES["default"]


class DegradationHandler:
    """
    Handles graceful degradation for each pipeline component.

    When a component fails, provides a user-friendly fallback that
    maintains as much functionality as possible.
    """

    def handle_asr_failure(
        self,
        error: Exception,
        latency: Any,
        total_start: float,
    ) -> Any:
        """
        Handle ASR failure by returning a text-input prompt.

        Args:
            error: The exception that caused ASR failure.
            latency: LatencyBreakdown instance.
            total_start: Pipeline start time.

        Returns:
            PipelineResult indicating ASR failure with guidance.
        """
        from .pipeline import PipelineResult, LatencyBreakdown

        logger.error("ASR failed: %s", error)

        latency.total_ms = (time.time() - total_start) * 1000

        return PipelineResult(
            query_text="",
            answer_text=(
                "I was unable to process the audio input. This could be due to "
                "audio quality, format, or a temporary processing issue. "
                "Please try again or type your question instead."
            ),
            audio_bytes=None,
            audio_format="",
            latency=latency,
            degraded=True,
            degradation_reason=f"ASR failure: {error}",
        )

    def handle_llm_failure(
        self,
        error: Exception,
        query: str,
        latency: Any,
        total_start: float,
    ) -> Any:
        """
        Handle LLM failure by returning a cached/fallback response.

        Args:
            error: The exception that caused LLM failure.
            query: The original user query.
            latency: LatencyBreakdown instance.
            total_start: Pipeline start time.

        Returns:
            PipelineResult with fallback answer.
        """
        from .pipeline import PipelineResult

        logger.error("LLM failed: %s", error)

        fallback = _find_fallback(query)
        latency.total_ms = (time.time() - total_start) * 1000

        return PipelineResult(
            query_text=query,
            answer_text=fallback,
            audio_bytes=None,
            audio_format="",
            latency=latency,
            model_used="fallback",
            degraded=True,
            degradation_reason=f"LLM failure: {error}",
        )

    def handle_tts_failure(
        self,
        error: Exception,
        query: str,
        answer_text: str,
        sources: list[str],
        model_used: str,
        latency: Any,
        total_start: float,
    ) -> Any:
        """
        Handle TTS failure by returning text-only response.

        Args:
            error: The exception that caused TTS failure.
            query: The original query.
            answer_text: The LLM-generated answer (still valid).
            sources: Source document references.
            model_used: LLM model that generated the answer.
            latency: LatencyBreakdown instance.
            total_start: Pipeline start time.

        Returns:
            PipelineResult with text answer but no audio.
        """
        from .pipeline import PipelineResult

        logger.error("TTS failed: %s", error)

        latency.total_ms = (time.time() - total_start) * 1000

        return PipelineResult(
            query_text=query,
            answer_text=answer_text,
            audio_bytes=None,
            audio_format="",
            latency=latency,
            sources=sources,
            model_used=model_used,
            degraded=True,
            degradation_reason=f"TTS failure: {error}. Text response provided instead.",
        )
