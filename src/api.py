"""
REST API for the Pharma Voice Assistant.

Provides endpoints for text queries, voice queries (audio file upload),
avatar video generation, health checks, and latency statistics.

Author: Gourav Pandey
"""

import io
import json
import logging
import os
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

from .avatar import HeyGenAvatar
from .llm_engine import LLMEngine
from .pipeline import VoicePipeline
from .tts import TTSFactory

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../frontend", static_url_path="/static")
CORS(app)

# Initialize components
pipeline = VoicePipeline()
llm_engine = LLMEngine()
avatar_client = HeyGenAvatar()

# Latency tracking (last 100 requests)
latency_history: deque[dict[str, Any]] = deque(maxlen=100)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Returns status of all pipeline components.
    """
    tts_available = TTSFactory.list_available()
    llm_available = llm_engine.list_available()

    return jsonify({
        "status": "healthy",
        "components": {
            "llm_providers": llm_available,
            "tts_engines": tts_available,
            "avatar_available": avatar_client.is_available(),
        },
    })


@app.route("/ask", methods=["POST"])
def ask_text():
    """
    Answer a text query with optional TTS audio.

    Request body:
        {
            "question": "What is the role of QA in batch release?",
            "include_audio": true (optional, default true),
            "provider": "ollama" (optional)
        }

    Response:
        {
            "answer": "...",
            "audio_url": "/audio/<id>" (if include_audio),
            "sources": [...],
            "latency": {...},
            "model": "..."
        }
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]
    include_audio = data.get("include_audio", True)

    try:
        result = pipeline.process_text(question)

        response_data = {
            "answer": result.answer_text,
            "sources": result.sources,
            "latency": result.latency.to_dict(),
            "model": result.model_used,
            "degraded": result.degraded,
        }

        # Track latency
        latency_history.append(result.latency.to_dict())

        if include_audio and result.audio_bytes:
            # Return audio as base64 in response
            import base64
            response_data["audio_base64"] = base64.b64encode(result.audio_bytes).decode()
            response_data["audio_format"] = result.audio_format

        return jsonify(response_data)

    except Exception as e:
        logger.error("Error processing text query: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ask/voice", methods=["POST"])
def ask_voice():
    """
    Answer a voice query (audio file upload).

    Accepts audio file via multipart form upload.
    Runs full pipeline: ASR -> RAG -> LLM -> TTS.

    Request:
        POST with multipart/form-data, 'audio' file field.

    Response:
        {
            "transcription": "...",
            "answer": "...",
            "audio_base64": "...",
            "latency": {...}
        }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file in request"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    # Determine format from filename
    filename = audio_file.filename or "audio.wav"
    audio_format = filename.rsplit(".", 1)[-1] if "." in filename else "wav"

    try:
        result = pipeline.process_voice(audio_bytes, audio_format=audio_format)

        response_data = {
            "transcription": result.query_text,
            "answer": result.answer_text,
            "sources": result.sources,
            "latency": result.latency.to_dict(),
            "model": result.model_used,
            "degraded": result.degraded,
        }

        latency_history.append(result.latency.to_dict())

        if result.audio_bytes:
            import base64
            response_data["audio_base64"] = base64.b64encode(result.audio_bytes).decode()
            response_data["audio_format"] = result.audio_format

        return jsonify(response_data)

    except Exception as e:
        logger.error("Error processing voice query: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ask/avatar", methods=["POST"])
def ask_avatar():
    """
    Generate an avatar video response.

    Request body:
        {
            "question": "...",
            "avatar_id": "default" (optional)
        }

    Response:
        {
            "answer": "...",
            "video_url": "...",
            "video_id": "...",
            "latency": {...}
        }
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    if not avatar_client.is_available():
        return jsonify({"error": "HeyGen avatar not configured. Set HEYGEN_API_KEY."}), 503

    question = data["question"]
    avatar_id = data.get("avatar_id")

    try:
        # Get LLM answer first
        llm_response = llm_engine.generate(question)

        # Generate avatar video
        avatar_result = avatar_client.generate_full(
            text=llm_response.text,
            avatar_id=avatar_id,
        )

        return jsonify({
            "answer": llm_response.text,
            "video_url": avatar_result.video_url,
            "video_id": avatar_result.video_id,
            "video_duration_s": avatar_result.duration_s,
            "latency": {
                "llm_ms": round(llm_response.latency_s * 1000, 1),
                "video_generation_ms": round(avatar_result.processing_time_s * 1000, 1),
            },
        })

    except Exception as e:
        logger.error("Error generating avatar response: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/latency-stats", methods=["GET"])
def latency_stats():
    """
    Return latency statistics from recent requests.

    Response includes P50 and P95 latencies for each pipeline stage.
    """
    if not latency_history:
        return jsonify({"message": "No requests processed yet", "stats": {}})

    stats: dict[str, dict[str, float]] = {}

    for key in ["asr_ms", "retrieval_ms", "llm_ms", "llm_ttft_ms", "tts_ms", "total_ms"]:
        values = sorted([h.get(key, 0) for h in latency_history if h.get(key, 0) > 0])
        if values:
            stats[key] = {
                "p50": values[len(values) // 2],
                "p95": values[int(len(values) * 0.95)],
                "min": values[0],
                "max": values[-1],
                "count": len(values),
            }

    return jsonify({
        "total_requests": len(latency_history),
        "stats": stats,
    })


@app.route("/")
def index():
    """Serve the demo frontend."""
    frontend_path = os.path.join(app.static_folder, "index.html")
    if os.path.exists(frontend_path):
        return send_file(frontend_path)
    return jsonify({"message": "Pharma Voice Assistant API. See /health for status."})


def create_app() -> Flask:
    """Application factory."""
    return app


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    logger.info("Starting Pharma Voice Assistant API on %s:%d", host, port)
    app.run(host=host, port=port, debug=False)
