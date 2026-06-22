"""Streaming read path — /tts/stream endpoint + CacheService.stream.

FakeProvider yields chunks; no network. Covers HIT (cached blob streamed),
MISS (provider chunks streamed + full clip stored on completion), and the
partial-consumption edge (no store when the consumer stops early).
"""

from __future__ import annotations

import pytest

from app.cache.service import CacheService
from app.core.config import settings
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _req() -> TTSRequest:
    return TTSRequest(
        model_id="cartesia:sonic-3.5",
        transcript="thank you",
        voice=CartesiaVoice(id="v1"),
        language="en",
        output_format=OutputFormat(),
    )


@pytest.fixture
async def svc(tmp_storage, fake_provider):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None)


# -- CacheService.stream -----------------------------------------------------


async def test_stream_miss_then_hit(svc, fake_provider):
    # MISS: provider streams chunks; the concatenated audio is the full clip.
    headers, gen = await svc.stream(_req())
    assert headers["X-Cache"] == "MISS"
    chunks = [c async for c in gen]
    assert b"".join(chunks) == fake_provider._audio
    assert fake_provider.stream_calls == 1

    # HIT: served from cache, provider not streamed again.
    headers2, gen2 = await svc.stream(_req())
    assert headers2["X-Cache"] == "HIT"
    chunks2 = [c async for c in gen2]
    assert b"".join(chunks2) == fake_provider._audio
    assert fake_provider.stream_calls == 1  # unchanged


async def test_stream_miss_stores_on_completion(svc, fake_provider):
    _, gen = await svc.stream(_req())
    # Fully consume the stream so the tee completes and stores.
    _ = [c async for c in gen]

    cached, record, *_ = await svc.check(_req())
    assert cached is True
    assert record.size_bytes == len(fake_provider._audio)


async def test_stream_partial_consumption_does_not_store(svc, fake_provider):
    """Stopping mid-stream must not cache a truncated clip."""
    _, gen = await svc.stream(_req())
    it = gen.__aiter__()
    await it.__anext__()  # take only the first chunk
    await gen.aclose()  # simulate client disconnect

    cached, *_ = await svc.check(_req())
    assert cached is False  # nothing stored
    assert fake_provider.stream_calls == 1


# -- /tts/stream endpoint ----------------------------------------------------


def test_endpoint_stream_miss_then_hit(app_client):
    fake = app_client._fake
    req = {
        "model_id": "cartesia:sonic-3.5",
        "transcript": "thank you",
        "voice": {"id": "v1"},
        "language": "en",
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
    }

    r1 = app_client.post("/tts/stream", json=req)
    assert r1.status_code == 200
    assert r1.headers["X-Cache"] == "MISS"
    body1 = r1.content
    assert body1 == fake._audio
    assert fake.stream_calls == 1

    r2 = app_client.post("/tts/stream", json=req)
    assert r2.status_code == 200
    assert r2.headers["X-Cache"] == "HIT"
    assert r2.content == body1
    assert fake.stream_calls == 1  # HIT served from cache


def test_endpoint_stream_bad_model_id(app_client):
    r = app_client.post(
        "/tts/stream",
        json={"model_id": "nope", "transcript": "x", "voice": {"id": "v1"}, "language": "en"},
    )
    assert r.status_code == 400
