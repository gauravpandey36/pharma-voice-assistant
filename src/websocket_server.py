"""
WebSocket server for real-time voice streaming.

Clients send audio chunks over WebSocket, server processes through
the voice pipeline, and streams back audio response chunks.

Author: Gourav Pandey
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

from .pipeline import VoicePipeline

logger = logging.getLogger(__name__)

WS_HOST = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("WS_PORT", "8001"))


class VoiceWebSocketServer:
    """
    WebSocket server for real-time voice interaction.

    Protocol:
    - Client sends JSON: {"type": "audio_start", "format": "wav"}
    - Client sends binary audio chunks
    - Client sends JSON: {"type": "audio_end"}
    - Server responds with JSON: {"type": "transcription", "text": "..."}
    - Server sends JSON: {"type": "answer", "text": "...", "latency": {...}}
    - Server sends binary audio response chunks
    - Server sends JSON: {"type": "audio_complete"}
    """

    def __init__(self, pipeline: VoicePipeline | None = None) -> None:
        """
        Initialize the WebSocket server.

        Args:
            pipeline: VoicePipeline instance. Created with defaults if None.
        """
        self._pipeline = pipeline or VoicePipeline()
        self._active_connections: set[WebSocketServerProtocol] = set()

    async def handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str = "/",
    ) -> None:
        """
        Handle a single WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            path: Connection path.
        """
        self._active_connections.add(websocket)
        client_id = id(websocket)
        logger.info("Client connected: %s (total: %d)", client_id, len(self._active_connections))

        audio_buffer = bytearray()
        audio_format = "wav"
        collecting_audio = False

        try:
            async for message in websocket:
                if isinstance(message, str):
                    # JSON control message
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "audio_start":
                            audio_buffer.clear()
                            audio_format = data.get("format", "wav")
                            collecting_audio = True
                            logger.debug("Client %s: audio collection started (%s)", client_id, audio_format)

                        elif msg_type == "audio_end":
                            collecting_audio = False
                            if audio_buffer:
                                await self._process_audio(
                                    websocket, bytes(audio_buffer), audio_format
                                )
                            audio_buffer.clear()

                        elif msg_type == "text_query":
                            # Direct text query (no ASR needed)
                            query = data.get("text", "")
                            if query:
                                await self._process_text(websocket, query)

                        elif msg_type == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))

                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON message",
                        }))

                elif isinstance(message, bytes):
                    # Binary audio data
                    if collecting_audio:
                        audio_buffer.extend(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected: %s", client_id)
        except Exception as e:
            logger.error("Error handling client %s: %s", client_id, e)
        finally:
            self._active_connections.discard(websocket)

    async def _process_audio(
        self,
        websocket: WebSocketServerProtocol,
        audio_bytes: bytes,
        audio_format: str,
    ) -> None:
        """
        Process collected audio through the voice pipeline.

        Args:
            websocket: Client WebSocket connection.
            audio_bytes: Collected audio bytes.
            audio_format: Audio format string.
        """
        logger.info("Processing %d bytes of audio (%s)", len(audio_bytes), audio_format)

        # Run pipeline in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._pipeline.process_voice,
            audio_bytes,
            audio_format,
        )

        # Send transcription
        await websocket.send(json.dumps({
            "type": "transcription",
            "text": result.query_text,
        }))

        # Send answer text
        await websocket.send(json.dumps({
            "type": "answer",
            "text": result.answer_text,
            "sources": result.sources,
            "model": result.model_used,
            "latency": result.latency.to_dict(),
            "degraded": result.degraded,
        }))

        # Send audio if available
        if result.audio_bytes:
            # Send in chunks for streaming
            chunk_size = 4096
            for i in range(0, len(result.audio_bytes), chunk_size):
                chunk = result.audio_bytes[i:i + chunk_size]
                await websocket.send(chunk)

            await websocket.send(json.dumps({
                "type": "audio_complete",
                "format": result.audio_format,
                "size_bytes": len(result.audio_bytes),
            }))
        else:
            await websocket.send(json.dumps({
                "type": "audio_unavailable",
                "reason": result.degradation_reason or "No audio generated",
            }))

    async def _process_text(
        self,
        websocket: WebSocketServerProtocol,
        query: str,
    ) -> None:
        """
        Process a text query through the pipeline (no ASR).

        Args:
            websocket: Client WebSocket connection.
            query: Text query.
        """
        logger.info("Processing text query: '%s'", query[:50])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._pipeline.process_text,
            query,
        )

        await websocket.send(json.dumps({
            "type": "answer",
            "text": result.answer_text,
            "sources": result.sources,
            "model": result.model_used,
            "latency": result.latency.to_dict(),
        }))

        if result.audio_bytes:
            chunk_size = 4096
            for i in range(0, len(result.audio_bytes), chunk_size):
                await websocket.send(result.audio_bytes[i:i + chunk_size])

            await websocket.send(json.dumps({
                "type": "audio_complete",
                "format": result.audio_format,
            }))

    async def start(self, host: str = WS_HOST, port: int = WS_PORT) -> None:
        """
        Start the WebSocket server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        logger.info("Starting WebSocket server on ws://%s:%d", host, port)

        async with websockets.serve(self.handle_connection, host, port):
            await asyncio.Future()  # Run forever


def main() -> None:
    """Start the WebSocket server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = VoiceWebSocketServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
