"""CacheService check/create/delete + read flow (FakeProvider, no network)."""

from __future__ import annotations

import asyncio

import pytest

from app.cache.service import CacheService
from app.core.config import settings
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _req() -> TTSRequest:
    return _req_text("thank you")


def _req_text(text: str) -> TTSRequest:
    return TTSRequest(
        model_id="cartesia:sonic-3.5",
        transcript=text,
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


async def test_check_before_after_create(svc, fake_provider):
    cached, record, *_ = await svc.check(_req())
    assert cached is False and record is None
    await svc.create(_req())
    cached, record, *_ = await svc.check(_req())
    assert cached is True and record is not None


async def test_create_override_resynthesizes(svc, fake_provider):
    await svc.create(_req())
    await svc.create(_req())
    assert fake_provider.calls == 2  # override re-synthesizes


async def test_create_from_base64_skips_provider(svc, fake_provider):
    audio = b"\x00\x01" * 50
    key, status, source, size, *_ = await svc.create(_req(), audio_override=audio)
    assert status == "CREATED" and source == "base64" and size == len(audio)
    assert fake_provider.calls == 0  # no provider call
    out, _ = await svc.get_or_synthesize(_req())
    assert out == audio  # stored verbatim, returned verbatim on hit


async def test_delete(svc):
    await svc.create(_req())
    deleted, _ = await svc.delete(_req())
    assert deleted is True
    deleted2, _ = await svc.delete(_req())
    assert deleted2 is False


async def test_miss_then_hit(svc, fake_provider):
    audio1, h1 = await svc.get_or_synthesize(_req())
    assert h1["X-Cache"] == "MISS"
    audio2, h2 = await svc.get_or_synthesize(_req())
    assert h2["X-Cache"] == "HIT"
    assert audio1 == audio2
    assert fake_provider.calls == 1


async def test_conversion_to_mulaw(svc, fake_provider):
    req = _req()
    req.output_format = OutputFormat(container="raw", encoding="mulaw", sample_rate=8000)
    audio, h = await svc.get_or_synthesize(req)
    assert h["X-Cache"] == "MISS"
    # 400 PCM frames @16k downsampled to 8k → 200 μ-law bytes
    assert len(audio) == pytest.approx(200, abs=4)


async def test_lru_no_stale_after_override(svc, fake_provider):
    a1, h1 = await svc.get_or_synthesize(_req())  # MISS, stores fake default
    assert h1["X-Cache"] == "MISS"
    a2, h2 = await svc.get_or_synthesize(_req())  # HIT, serves cached (== a1)
    assert h2["X-Cache"] == "HIT" and a2 == a1

    override = bytes([0xAB]) * len(a1)
    await svc.create(_req(), audio_override=override)  # override updates LRU
    a3, h3 = await svc.get_or_synthesize(_req())  # HIT but now overridden bytes
    assert h3["X-Cache"] == "HIT" and a3 == override and a3 != a1


async def test_delete_then_read_resynthesizes(svc, fake_provider):
    await svc.create(_req())  # synth (calls=1)
    await svc.get_or_synthesize(_req())  # HIT (no synth)
    await svc.delete(_req())
    audio, h = await svc.get_or_synthesize(_req())  # MISS again — no stale serve
    assert h["X-Cache"] == "MISS"
    assert fake_provider.calls == 2  # re-synthesized after delete


async def test_unified_cache_one_entry_serves_both_formats(svc, fake_provider):
    """Key is format-agnostic: warming via the mulaw one-shot path also serves
    the pcm streaming path (and vice versa) — one synth, one entry."""
    mulaw_req = _req()
    mulaw_req.output_format = OutputFormat(container="raw", encoding="mulaw", sample_rate=8000)
    pcm_req = _req()  # default pcm_s16le @ 16k

    # Warm via the one-shot μ-law path (stores native pcm under a format-agnostic key).
    await svc.create(mulaw_req)
    assert fake_provider.calls == 1

    # Streaming the SAME phrase as pcm@16k is a HIT — no re-synth.
    headers, gen = await svc.stream(pcm_req)
    assert headers["X-Cache"] == "HIT"
    chunks = [c async for c in gen]
    assert b"".join(chunks) == fake_provider._audio  # served as native pcm
    assert fake_provider.calls == 1

    # And the one-shot μ-law path hits the same entry (converts native→mulaw on serve).
    audio, h = await svc.get_or_synthesize(mulaw_req)
    assert h["X-Cache"] == "HIT"
    assert fake_provider.calls == 1


async def test_stitch_serves_miss_from_cached_subphrases(svc, fake_provider, monkeypatch):
    """Full-text MISS is stitched from cached sub-phrases + a synthesized gap."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    for w in ("hi", "sir"):  # pre-seed the fixed parts
        await svc.create(_req_text(w))
    seed_calls = fake_provider.calls  # 2

    audio, h = await svc.get_or_synthesize(_req_text("hi nitya sir"))  # MISS -> stitch
    assert h["X-Cache"] == "MISS-STITCH"
    # only the gap "nitya" was synthesized; "hi"/"sir" served from cache
    assert fake_provider.calls == seed_calls + 1
    assert len(audio) > 0


async def test_stitch_skipped_when_coverage_below_gate(svc, fake_provider, monkeypatch):
    """Too little cached -> fall back to a full synth (no stitch)."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    await svc.create(_req_text("hi"))  # only 1 of 5 words cached
    audio, h = await svc.get_or_synthesize(_req_text("hi big unknown phrase here"))
    assert h["X-Cache"] == "MISS"  # full synth, not stitched (coverage 1/5 < 0.5)


async def test_stream_serves_miss_stitch_then_hit(svc, fake_provider, monkeypatch):
    """Streaming MISS is stitched from cached sub-phrases, stored, then re-served
    as a HIT — so warmed sub-phrases aren't dead weight on /tts/stream."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    monkeypatch.setattr(settings, "predictive_stitch_stream_enabled", True)
    for w in ("hi", "sir"):
        await svc.create(_req_text(w))
    seed_calls = fake_provider.calls

    headers, gen = await svc.stream(_req_text("hi nitya sir"))
    assert headers["X-Cache"] == "MISS-STITCH"
    chunks = [c async for c in gen]
    assert len(b"".join(chunks)) > 0
    assert fake_provider.calls == seed_calls + 1  # only the gap "nitya" synthesized

    # The assembled clip was stored -> a repeat is an instant HIT.
    h2, _gen2 = await svc.stream(_req_text("hi nitya sir"))
    assert h2["X-Cache"] == "HIT"
    assert fake_provider.calls == seed_calls + 1  # no further synth


def test_stitch_clips_removes_dc_and_silence():
    """Single-clip path: DC offset removed and edge silence trimmed."""
    import numpy as np
    from app.cache.service import _stitch_clips, _to_int16, _to_float

    sr = 16_000
    tone = 0.4 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr) + 0.3  # + DC bias
    padded = np.concatenate(
        [np.zeros(sr // 4), tone, np.zeros(sr // 4)]
    ).astype(np.float32)
    out = _to_float(_stitch_clips([_to_int16(padded)]))
    assert len(out) < len(padded) - sr // 4          # leading/trailing silence gone
    assert abs(float(np.mean(out))) < 0.02           # DC removed


def test_stitch_clips_equalizes_loudness():
    """A quiet clip and a loud clip come out at comparable loudness."""
    import numpy as np
    from app.cache.service import _stitch_clips, _to_int16, _to_float

    sr = 16_000
    t = np.arange(sr // 2) / sr
    quiet = (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # RMS ~0.035
    loud = (0.6 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)    # RMS ~0.42
    raw = _stitch_clips([_to_int16(quiet), _to_int16(loud)])
    assert len(raw) % 2 == 0 and len(raw) > 0   # valid int16 byte stream
    out = _to_float(raw)
    half = len(out) // 2
    r_quiet = float(np.sqrt(np.mean(out[:half] ** 2)))
    r_loud = float(np.sqrt(np.mean(out[half : half * 2] ** 2)))
    assert r_quiet > 0.05                            # quiet clip was amplified up
    assert max(r_quiet, r_loud) / max(1e-6, min(r_quiet, r_loud)) < 4.0  # roughly matched


# -- single-flight + totals correctness ------------------------------------


async def test_single_flight_concurrent_misses_share_one_synth(svc, fake_provider):
    """N concurrent identical MISSes share ONE synth + ONE store (single-flight):
    the producer records a MISS, coalesced callers record HITs, and the provider
    is hit exactly once."""
    req = _req_text("the quick brown fox jumps")
    results = await asyncio.gather(*(svc.get_or_synthesize(req) for _ in range(8)))
    assert fake_provider.calls == 1  # one synth despite 8 concurrent misses
    audios = [a for a, _ in results]
    assert all(a == audios[0] for a in audios) and len(audios[0]) > 0  # identical audio
    statuses = [h["X-Cache"] for _, h in results]
    assert statuses.count("MISS") == 1   # the producer
    assert statuses.count("HIT") == 7    # coalesced onto the in-flight synth


async def test_put_with_totals_concurrent_same_key_no_drift(tmp_storage):
    """Concurrent fresh stores of the SAME key bump provider_totals exactly once
    (per-worker-thread connections + INSERT OR IGNORE + rowcount), so /stats
    never drifts under contention. This is the test that exposed the old
    shared-connection concurrency bug."""
    from app.storage.base import CacheRecord
    from app.storage.sqlite import SQLiteMetadataStore

    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    rec = CacheRecord(
        key="k1", provider="cartesia", voice_id="v", model="m", language="en",
        params="", text="t", container="raw", encoding="pcm_s16le", sample_rate=16000,
        size_bytes=100, storage_path="ab/cd/k1", hit_count=0,
        created_at="2026-01-01T00:00:00+00:00", last_accessed_at="2026-01-01T00:00:00+00:00",
        ttl_expires_at=None,
    )
    await asyncio.gather(*(meta.put_with_totals(rec) for _ in range(16)))
    snap = await meta.stats()
    assert snap["entries"] == 1            # one row, not 16
    assert snap["total_bytes"] == 100      # counted once, not 1600


async def test_concurrent_distinct_writes_no_loss(tmp_storage):
    """Many concurrent stores of DISTINCT keys all land (no lost writes) and
    totals stay exact — the realistic 'N parallel misses on different phrases'
    shape. Proves the per-worker-thread connection model is safe under load."""
    from app.storage.base import CacheRecord
    from app.storage.sqlite import SQLiteMetadataStore

    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()

    def rec(i: int) -> CacheRecord:
        return CacheRecord(
            key=f"k{i}", provider="cartesia", voice_id="v", model="m", language="en",
            params="", text="t", container="raw", encoding="pcm_s16le", sample_rate=16000,
            size_bytes=10, storage_path=f"ab/cd/k{i}", hit_count=0,
            created_at="2026-01-01T00:00:00+00:00",
            last_accessed_at="2026-01-01T00:00:00+00:00", ttl_expires_at=None,
        )

    # 60 concurrent writes of distinct keys + 60 concurrent reads of the same.
    await asyncio.gather(*(meta.put_with_totals(rec(i)) for i in range(60)))
    got = await asyncio.gather(*(meta.get(f"k{i}") for i in range(60)))
    snap = await meta.stats()
    assert snap["entries"] == 60                 # none lost
    assert snap["total_bytes"] == 60 * 10
    assert all(r is not None for r in got)       # every key readable

