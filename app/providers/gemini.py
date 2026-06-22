"""Gemini TTS provider adapter.

Ports clairvoyance's ``_generate_gemini_audio`` helper into the uniform
``BaseTTSProvider.synth`` contract. On a cache miss DragonTTS delegates raw
synthesis here, then a separate conversion layer maps the native format to the
caller's requested ``output_format`` before caching.

Gemini TTS outputs 24 kHz PCM natively. The reference clairvoyance helper
downsamples to 16 kHz PCM before returning; this adapter reproduces that exact
behavior so the returned bytes are 16 kHz raw PCM.
"""

from __future__ import annotations

import json

import numpy as np
from google.cloud import texttospeech_v1
from google.oauth2 import service_account

from app.core.config import PROVIDER_DEFAULTS, settings
from app.core.logging import logger
from app.providers.base import AudioResult, BaseTTSProvider

# Gemini TTS native sample rate — fixed by the API.
_GEMINI_SAMPLE_RATE = 24_000


class GeminiProvider(BaseTTSProvider):
    """Gemini TTS backend.

    Returns audio in Gemini's native raw PCM format (16-bit signed LE),
    downsampled from the API's native 24 kHz to 16 kHz mono — mirroring
    clairvoyance's ``_generate_gemini_audio``.
    """

    name = "gemini"

    def __init__(
        self,
        credentials_json: str | None = None,
        credentials_path: str | None = None,
    ):
        """Initialize the provider.

        Args:
            credentials_json: GCP service-account JSON string. Defaults to
                ``settings.google_credentials_json`` when ``None``.
            credentials_path: Path to a GCP service-account JSON file. Defaults
                to ``settings.google_credentials_path`` when ``None``.
        """
        self.credentials_json = credentials_json or settings.google_credentials_json
        self.credentials_path = credentials_path or settings.google_credentials_path

    async def synth(
        self,
        *,
        text: str,
        voice_id: str | None,
        model: str | None,
        language: str | None,
        params: dict,
    ) -> AudioResult:
        """Synthesize ``text`` and return native-format raw PCM audio.

        Authenticates with a GCP service account (JSON string or file path)
        and calls the Gemini TTS streaming API, then downsamples the native
        24 kHz PCM to 16 kHz — matching clairvoyance's reference helper.

        Fields missing from the request fall back to
        ``PROVIDER_DEFAULTS["gemini"]``.

        Args:
            text: The text to synthesize.
            voice_id: Gemini voice name (e.g. "Kore") override, or ``None``/
                empty for the default.
            model: Gemini TTS model override, or ``None``/empty for the
                default.
            language: BCP-47 language code (e.g. "en-IN") override, or
                ``None``/empty for the default.
            params: Reserved for future provider-specific options. ``style_prompt``
                (a natural-language style instruction) is honored if present.

        Returns:
            AudioResult holding raw ``pcm_s16le`` audio at 16 kHz.

        Raises:
            ValueError: If neither ``credentials_json`` nor ``credentials_path``
                is configured.
        """
        if not self.credentials_json and not self.credentials_path:
            raise ValueError(
                "GOOGLE_CREDENTIALS_JSON/PATH is required for Gemini TTS"
            )

        defaults = PROVIDER_DEFAULTS["gemini"]
        final_voice_id = voice_id or defaults.get("voice_id", "Kore")
        final_model = model or defaults.get("model", "")
        final_language = language or defaults.get("language", "en-IN")
        style_prompt = params.get("style_prompt") if params else None

        logger.info(
            f"Synthesizing audio with Gemini: {text[:50]}... "
            f"[voice_id={final_voice_id}, model={final_model}, lang={final_language}]"
        )

        # Build authenticated client — GCP service-account JSON (string or file).
        if self.credentials_json:
            json_account_info = json.loads(self.credentials_json)
            creds = service_account.Credentials.from_service_account_info(
                json_account_info
            )
        else:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
        client = texttospeech_v1.TextToSpeechAsyncClient(credentials=creds)

        try:
            # Voice params — model_name routes to the Gemini TTS model.
            voice_params = texttospeech_v1.VoiceSelectionParams(
                language_code=final_language,
                name=final_voice_id,
                model_name=final_model,
            )

            streaming_config = texttospeech_v1.StreamingSynthesizeConfig(
                voice=voice_params,
                streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                    audio_encoding=texttospeech_v1.AudioEncoding.PCM,
                    sample_rate_hertz=_GEMINI_SAMPLE_RATE,
                ),
            )

            async def _request_generator():
                yield texttospeech_v1.StreamingSynthesizeRequest(
                    streaming_config=streaming_config
                )
                synthesis_params: dict = {"text": text}
                if style_prompt:
                    synthesis_params["prompt"] = style_prompt
                yield texttospeech_v1.StreamingSynthesizeRequest(
                    input=texttospeech_v1.StreamingSynthesisInput(**synthesis_params)
                )

            chunks: list[bytes] = []
            streaming_responses = await client.streaming_synthesize(_request_generator())
            async for response in streaming_responses:
                if response.audio_content:
                    chunks.append(response.audio_content)

            pcm_24k = b"".join(chunks)

            if not pcm_24k:
                raise RuntimeError(
                    "Gemini TTS returned empty audio — check credentials and voice settings"
                )

            # Ensure whole-frame alignment before resampling (16-bit = 2 bytes/frame).
            if len(pcm_24k) % 2 != 0:
                pcm_24k += b"\x00"

            # Downsample 24 kHz -> 16 kHz via linear interpolation (no scipy needed).
            # 24000 * (2/3) = 16000.
            samples = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32)
            out_len = len(samples) * 16_000 // _GEMINI_SAMPLE_RATE
            indices = np.linspace(0, len(samples) - 1, out_len)
            lo = np.floor(indices).astype(np.int32)
            hi = np.minimum(lo + 1, len(samples) - 1)
            frac = (indices - lo).astype(np.float32)
            resampled = samples[lo] + frac * (samples[hi] - samples[lo])
            pcm_16k = np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

            return AudioResult(
                audio=pcm_16k,
                container="raw",
                encoding="pcm_s16le",
                sample_rate=16000,
            )
        finally:
            await client.transport.grpc_channel.close()  # type: ignore[union-attr]
