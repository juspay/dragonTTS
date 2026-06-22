"""Sarvam TTS provider adapter.

Ports clairvoyance's ``_generate_sarvam_audio`` helper into the uniform
:class:`BaseTTSProvider` contract. Sarvam returns raw 16-bit PCM at 16 kHz
(base64-encoded inside its JSON response), so the native output format reported
here is ``pcm_s16le`` at 16000 Hz.
"""

from __future__ import annotations

import base64

import httpx

from app.core.config import PROVIDER_DEFAULTS, settings
from app.core.logging import logger
from app.providers.base import AudioResult, BaseTTSProvider

# Sarvam always returns 16-bit PCM at 16 kHz.
_SARVAM_SAMPLE_RATE = 16000


class SarvamProvider(BaseTTSProvider):
    """Adapter for the Sarvam text-to-speech HTTP API."""

    name = "sarvam"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.sarvam_api_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def synth(
        self,
        *,
        text: str,
        voice_id: str,
        model: str | None,
        language: str | None,
        params: dict,
    ) -> AudioResult:
        """Synthesize ``text`` and return raw 16-bit PCM at 16 kHz.

        Missing fields fall back to ``PROVIDER_DEFAULTS["sarvam"]``. The request
        mirrors clairvoyance's ``_generate_sarvam_audio``: same endpoint,
        ``api-subscription-key`` header, and JSON payload.
        """
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required")

        defaults = PROVIDER_DEFAULTS["sarvam"]
        voice = voice_id or defaults.get("voice_id", "shreya")
        model = model or defaults.get("model", "bulbul:v3")
        language_code = language or defaults.get("language", "en-IN")
        params = params or {}

        # speed/pace and pitch may arrive either in ``params`` or via defaults.
        pitch = params.get("pitch")
        if pitch is None:
            pitch = defaults.get("pitch", 0.0)
        pace = params.get("pace")
        if pace is None:
            pace = params.get("speed")
        if pace is None:
            pace = defaults.get("speed", 0.9)

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": [text],
            "target_language_code": language_code,
            "speaker": voice,
            "pitch": pitch,
            "pace": pace,
            "loudness": 1.5,
            "speech_sample_rate": _SARVAM_SAMPLE_RATE,
            "enable_preprocessing": True,
            "model": model,
        }

        logger.info(f"Synthesizing with Sarvam: {text[:50]}...")

        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        audio_base64 = result.get("audios", [None])[0]
        if not audio_base64:
            raise Exception("No audio returned from Sarvam API")

        audio = base64.b64decode(audio_base64)

        return AudioResult(
            audio=audio,
            container="raw",
            encoding="pcm_s16le",
            sample_rate=_SARVAM_SAMPLE_RATE,
        )
