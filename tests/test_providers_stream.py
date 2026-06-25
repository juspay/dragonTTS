"""Provider stream_synth fallback wiring (no network/keys).

When pooling is disabled (pool size 0) or no warm socket is available, a
provider's ``stream_synth`` must fall back to the one-shot ``synth`` path and
still yield audio — so a miss never fails just because the WS pool is off. These
monkeypatch ``synth`` to a stub (no provider API call) and assert the fallback
yields the synth output byte-for-byte.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.providers.base import AudioResult


async def _drain(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return b"".join(out)


async def test_elevenlabs_stream_falls_back_to_synth_when_pool_off(monkeypatch):
    from app.providers.elevenlabs import ElevenLabsProvider

    monkeypatch.setattr(settings, "elevenlabs_stream_pool_size", 0)
    prov = ElevenLabsProvider(api_key="k", base_url="https://api.elevenlabs.io")

    payload = b"\x01\x02\x03\x04" * 10
    async def fake_synth(*, text, voice_id, model, language, params):
        assert text == "hello"
        return AudioResult(payload, "raw", "pcm_s16le", 16000)

    monkeypatch.setattr(prov, "synth", fake_synth)
    out = await _drain(prov.stream_synth(
        text="hello", voice_id="v1", model=None, language=None, params={},
    ))
    assert out == payload


async def test_sarvam_stream_falls_back_to_synth_when_pool_off(monkeypatch):
    from app.providers.sarvam import SarvamProvider

    monkeypatch.setattr(settings, "sarvam_stream_pool_size", 0)
    prov = SarvamProvider(api_key="k")

    payload = b"\x05\x06\x07\x08" * 10
    async def fake_synth(*, text, voice_id, model, language, params):
        assert text == "namaste"
        return AudioResult(payload, "raw", "pcm_s16le", 16000)

    monkeypatch.setattr(prov, "synth", fake_synth)
    out = await _drain(prov.stream_synth(
        text="namaste", voice_id="shreya", model=None, language=None, params={},
    ))
    assert out == payload


async def test_elevenlabs_missing_key_raises(monkeypatch):
    from app.providers.elevenlabs import ElevenLabsProvider
    from app.providers.base import ProviderError

    monkeypatch.setattr(settings, "elevenlabs_stream_pool_size", 0)
    prov = ElevenLabsProvider(api_key="", base_url="https://api.elevenlabs.io")
    with pytest.raises(ProviderError):
        async for _ in prov.stream_synth(
            text="x", voice_id="v", model=None, language=None, params={},
        ):
            pass
