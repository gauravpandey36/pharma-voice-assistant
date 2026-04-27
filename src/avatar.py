"""
HeyGen avatar video integration module.

Creates talking-head avatar videos from text, with retry logic
and status polling for asynchronous video generation.

Author: Gourav Pandey
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

HEYGEN_API_BASE = "https://api.heygen.com/v2"
DEFAULT_API_KEY = os.environ.get("HEYGEN_API_KEY", "")


@dataclass
class AvatarVideoResult:
    """Result from avatar video generation."""

    video_id: str
    status: str
    video_url: str | None
    duration_s: float
    processing_time_s: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "video_id": self.video_id,
            "status": self.status,
            "video_url": self.video_url,
            "duration_s": round(self.duration_s, 2),
            "processing_time_s": round(self.processing_time_s, 2),
        }


class HeyGenAvatar:
    """
    HeyGen avatar video generation client.

    Handles video creation, status polling, and download with
    configurable retry logic and timeout handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        avatar_id: str = "default",
        max_retries: int = 3,
        poll_interval_s: float = 5.0,
        timeout_s: float = 300.0,
    ) -> None:
        """
        Initialize the HeyGen avatar client.

        Args:
            api_key: HeyGen API key. Defaults to HEYGEN_API_KEY env var.
            avatar_id: Default avatar ID to use.
            max_retries: Maximum retry attempts for API calls.
            poll_interval_s: Seconds between status polls.
            timeout_s: Maximum seconds to wait for video generation.
        """
        self._api_key = api_key or DEFAULT_API_KEY
        self._avatar_id = avatar_id
        self._max_retries = max_retries
        self._poll_interval = poll_interval_s
        self._timeout = timeout_s
        self._session = requests.Session()
        self._session.headers.update({
            "X-Api-Key": self._api_key,
            "Content-Type": "application/json",
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make an API request with retry logic.

        Args:
            method: HTTP method ('GET', 'POST').
            endpoint: API endpoint path.
            **kwargs: Additional arguments for requests.

        Returns:
            Response JSON.

        Raises:
            requests.HTTPError: If all retries fail.
        """
        url = f"{HEYGEN_API_BASE}/{endpoint}"

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.request(method, url, timeout=30, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "HeyGen API request failed (attempt %d/%d): %s",
                    attempt, self._max_retries, e,
                )
                if attempt == self._max_retries:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return {}  # Should not reach here

    def create_video(
        self,
        text: str,
        avatar_id: str | None = None,
        voice_id: str | None = None,
    ) -> str:
        """
        Create a talking-head avatar video.

        Args:
            text: Text for the avatar to speak.
            avatar_id: Override avatar ID (uses default if None).
            voice_id: Voice ID for speech synthesis.

        Returns:
            Video generation task ID.
        """
        avatar = avatar_id or self._avatar_id

        payload = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": avatar,
                        "avatar_style": "normal",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": text,
                    },
                }
            ],
            "dimension": {"width": 1280, "height": 720},
        }

        if voice_id:
            payload["video_inputs"][0]["voice"]["voice_id"] = voice_id

        logger.info("Creating avatar video: %d chars, avatar=%s", len(text), avatar)
        result = self._make_request("POST", "video/generate", json=payload)

        video_id = result.get("data", {}).get("video_id", "")
        if not video_id:
            raise ValueError(f"Failed to create video: {result}")

        logger.info("Video creation initiated: %s", video_id)
        return video_id

    def check_status(self, video_id: str) -> dict[str, Any]:
        """
        Check the status of a video generation task.

        Args:
            video_id: The video generation task ID.

        Returns:
            Status dictionary with 'status' and optional 'video_url' keys.
        """
        result = self._make_request("GET", f"video_status.get?video_id={video_id}")
        data = result.get("data", {})

        return {
            "video_id": video_id,
            "status": data.get("status", "unknown"),
            "video_url": data.get("video_url"),
            "duration": data.get("duration", 0),
            "error": data.get("error"),
        }

    def wait_for_completion(self, video_id: str) -> dict[str, Any]:
        """
        Poll until video generation is complete or timeout.

        Args:
            video_id: The video generation task ID.

        Returns:
            Final status dictionary.

        Raises:
            TimeoutError: If video generation exceeds timeout.
        """
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > self._timeout:
                raise TimeoutError(
                    f"Video generation timed out after {self._timeout}s for {video_id}"
                )

            status = self.check_status(video_id)
            current_status = status.get("status", "")

            if current_status == "completed":
                logger.info("Video generation complete: %s (%.1fs)", video_id, elapsed)
                return status
            elif current_status in ("failed", "error"):
                error = status.get("error", "Unknown error")
                logger.error("Video generation failed: %s - %s", video_id, error)
                raise RuntimeError(f"Video generation failed: {error}")

            logger.debug("Video %s status: %s (%.1fs elapsed)", video_id, current_status, elapsed)
            time.sleep(self._poll_interval)

    def download_video(
        self,
        video_url: str,
        output_path: str | Path,
    ) -> Path:
        """
        Download a completed video from its URL.

        Args:
            video_url: URL of the generated video.
            output_path: Local path to save the video.

        Returns:
            Path to the downloaded video file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading video to %s", output_path)
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Video downloaded: %s (%d bytes)", output_path, output_path.stat().st_size)
        return output_path

    def generate_full(
        self,
        text: str,
        output_path: str | Path | None = None,
        avatar_id: str | None = None,
    ) -> AvatarVideoResult:
        """
        Complete video generation pipeline: create, wait, download.

        Args:
            text: Text for the avatar to speak.
            output_path: Optional path to save the video file.
            avatar_id: Override avatar ID.

        Returns:
            AvatarVideoResult with video details.
        """
        start = time.time()

        # Create
        video_id = self.create_video(text, avatar_id=avatar_id)

        # Wait
        status = self.wait_for_completion(video_id)
        video_url = status.get("video_url", "")
        duration = status.get("duration", 0)

        # Download if path provided
        if output_path and video_url:
            self.download_video(video_url, output_path)

        processing_time = time.time() - start

        return AvatarVideoResult(
            video_id=video_id,
            status="completed",
            video_url=video_url,
            duration_s=float(duration),
            processing_time_s=processing_time,
        )

    def is_available(self) -> bool:
        """Check if the HeyGen API is configured."""
        return bool(self._api_key)
