"""ElevenLabs TTS provider adapter.

Ports clairvoyance's ``_generate_elevenlabs_audio`` helper into the uniform
``BaseTTSProvider`` contract. The registry constructs two instances of this
class — one for the global API (``elevenlabs``) and one for the Indian
residency API (``elevenlabs-in``) — differing only in ``api_key``/``base_url``.

Native output format: raw ulaw 8 kHz bytes (ElevenLabs ``output_format=ulaw_8000``).
A separate conversion layer (app/audio/format.py) maps this to the caller's
requested ``output_format`` before caching.
"""

from __future__ import annotations

import httpx

from app.core.config import PROVIDER_DEFAULTS, settings
from app.core.logging import logger
from app.providers.base import AudioResult, BaseTTSProvider


class ElevenLabsProvider(BaseTTSProvider):
    """ElevenLabs text-to-speech adapter.

    A single instance speaks to one ElevenLabs deployment (global or Indian
    residency); the registry wires the correct credentials per route.
    """

    name = "elevenlabs"
    # ElevenLabs returns μ-law @ 8 kHz natively (matches the telephony output).
    native_encoding = "ulaw"
    native_sample_rate = 8000

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize the provider.

        Args:
            api_key: ElevenLabs API key. Defaults to ``settings.elevenlabs_api_key``.
            base_url: ElevenLabs API base URL. Defaults to
                ``settings.elevenlabs_base_url``.
        """
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.base_url = (
            base_url if base_url is not None else settings.elevenlabs_base_url
        )
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
        """Synthesize ``text`` and return raw ulaw 8 kHz audio.

        Args:
            text: The text to synthesize.
            voice_id: ElevenLabs voice ID. Falls back to
                ``PROVIDER_DEFAULTS["elevenlabs"]["voice_id"]`` when empty.
            model: ElevenLabs model ID. Falls back to the provider default when
                empty.
            language: BCP-47 language hint. Falls back to the provider default
                when empty.
            params: Extra provider-specific parameters (reserved for future use).

        Returns:
            AudioResult wrapping the provider's native ulaw_8000 bytes.

        Raises:
            ValueError: If ``self.api_key`` is missing.
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required")

        defaults = PROVIDER_DEFAULTS["elevenlabs"]
        final_voice_id = voice_id if voice_id else defaults["voice_id"]
        final_model_id = model if model else defaults["model"]
        final_language = language if language else defaults["language"]

        url = f"{self.base_url}/v1/text-to-speech/{final_voice_id}?output_format=ulaw_8000"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/basic",
        }
        # NOTE: language_code is intentionally NOT sent — the ElevenLabs
        # /v1/text-to-speech/{voice_id} body has no such field (language is
        # bound to the model), and clairvoyance's reference doesn't send it.
        payload = {
            "text": text,
            "model_id": final_model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        logger.info(
            f"Synthesizing with ElevenLabs (ulaw_8000): {text[:50]}... "
            f"[voice_id={final_voice_id}, model_id={final_model_id}, "
            f"language={final_language}, base_url={self.base_url}]"
        )

        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return AudioResult(
            audio=response.content,
            container="raw",
            encoding="ulaw",
            sample_rate=8000,
        )
