"""CacheService check/create/delete + read flow (FakeProvider, no network)."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.cache.service import CacheService
from app.core.config import settings
from app.providers.base import AudioResult
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


async def test_stitch_uses_middle_cached_phrase(svc, fake_provider, monkeypatch):
    """A cached phrase in the MIDDLE of the request is reused. The old binary
    search only probed cached prefix/suffix edges (needed substring closure to
    reach a middle span); the DP segmentation finds it directly from an exact
    whole-phrase cache."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    await svc.create(_req_text("your order"))  # cached, sits in the middle
    seed_calls = fake_provider.calls
    audio, h = await svc.get_or_synthesize(_req_text("hi your order now"))
    assert h["X-Cache"] == "MISS-STITCH"
    # only the two gaps ("hi", "now") synthesized; "your order" served from cache
    assert fake_provider.calls == seed_calls + 2
    assert len(audio) > 0


class _RecordingElevenLabs:
    """ElevenLabs stand-in: deterministic PCM; records every synth text + count."""

    name = "elevenlabs"
    native_encoding = "pcm_s16le"
    native_sample_rate = 16000

    def __init__(self):
        self._audio = b"\x01\x00" * 400
        self.calls = 0
        self.seen: list[str] = []

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        self.calls += 1
        self.seen.append(text)
        return AudioResult(self._audio, "raw", "pcm_s16le", 16000)


async def test_stitch_elevenlabs_dot_keys_match_without_dot(tmp_storage, monkeypatch):
    """Regression: the cache KEY is dot-free, so a non-prefix cached ElevenLabs
    phrase is reused by stitch even though the SYNTH text carries the leading
    dot. An earlier version baked the dot into the key, so the middle cached
    phrase's stored key ('.your order') never matched the sub-span candidate
    ('your order') and stitch silently re-synthesized everything.

    Uses the prod config: dot ON, number normalization OFF (the dot is now
    independent of normalize)."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    monkeypatch.setattr(settings, "tts_normalize_numbers", False)
    monkeypatch.setattr(settings, "tts_leading_dot", True)

    prov = _RecordingElevenLabs()
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: prov if name == "elevenlabs" else None)

    def _req(text: str) -> TTSRequest:
        return TTSRequest(
            model_id="elevenlabs:eleven_flash_v2_5", transcript=text,
            voice=CartesiaVoice(id="v1"), language="en", output_format=OutputFormat(),
        )

    # Seed the MIDDLE phrase -> cached under a DOT-FREE key, but synthesized
    # with the ElevenLabs leading dot.
    await svc.create(_req("your order"))
    seed_calls = prov.calls
    assert prov.seen[-1] == ".your order"   # dot reached the provider

    # Full-text MISS: stitch must reuse "your order" (dot-free key match) and
    # synthesize only the two gaps ("hi", "now") -- not the whole phrase.
    audio, h = await svc.get_or_synthesize(_req("hi your order now"))
    assert h["X-Cache"] == "MISS-STITCH"
    assert prov.calls == seed_calls + 2          # only the gaps synthesized
    assert len(audio) > 0
    # The gap synths also carried the dot -> consistent onset/prosody with the
    # cached clip (the fix for the seam inconsistency the review flagged).
    assert ".hi" in prov.seen and ".now" in prov.seen


async def test_stitch_falls_through_on_blob_eviction(svc, fake_provider, monkeypatch):
    """A cached span whose blob vanished between the probe and the fetch is
    synthesized, not raised as a 500. Metadata delete + blob delete aren't atomic
    across the two stores, so the blob can be missing while the row still reads
    non-expired."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    key, *_ = await svc.create(_req_text("your order"))  # cached middle phrase
    seed_calls = fake_provider.calls
    rec = await svc._metadata.get(key)
    # Delete the BLOB but leave the metadata row (the eviction race).
    (svc._blobs.blob_dir / rec.storage_path).unlink()

    # Must NOT 500: the evicted cached span falls through to a synth.
    audio, h = await svc.get_or_synthesize(_req_text("hi your order now"))
    assert h["X-Cache"] == "MISS-STITCH"
    assert len(audio) > 0
    # 2 gaps ("hi", "now") + the evicted "your order" span, all synthesized.
    assert fake_provider.calls == seed_calls + 3


# -- ElevenLabs dot decoupling (synth-only, never the key) ------------------


async def _eleven_svc(prov) -> CacheService:
    """CacheService wired to an ElevenLabs provider for dot-decoupling tests."""
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: prov if name == "elevenlabs" else None)


def _eleven_req(text: str) -> TTSRequest:
    return TTSRequest(
        model_id="elevenlabs:eleven_flash_v2_5", transcript=text,
        voice=CartesiaVoice(id="v1"), language="en", output_format=OutputFormat(),
    )


async def test_dot_toggle_does_not_split_cache_entry(tmp_storage, monkeypatch):
    """TTS_LEADING_DOT is synth-only: flipping it must NOT change the cache key,
    so 'your order' stays a single entry whether the dot is on or off."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", False)
    prov = _RecordingElevenLabs()
    svc = await _eleven_svc(prov)

    monkeypatch.setattr(settings, "tts_leading_dot", True)
    await svc.create(_eleven_req("your order"))     # stored under a dot-free key
    calls_with_dot = prov.calls

    # Turn the dot OFF and request the SAME phrase -> same key, no re-synth.
    monkeypatch.setattr(settings, "tts_leading_dot", False)
    _audio, h = await svc.get_or_synthesize(_eleven_req("your order"))
    assert h["X-Cache"] == "HIT"
    assert prov.calls == calls_with_dot              # dot toggle didn't split


async def test_dot_reaches_synth_even_with_normalize_off(tmp_storage, monkeypatch):
    """The dot is INDEPENDENT of number normalization: with normalize OFF and dot
    ON, ElevenLabs still receives the leading dot (digits left unexpanded)."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", False)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    prov = _RecordingElevenLabs()
    svc = await _eleven_svc(prov)

    await svc.get_or_synthesize(_eleven_req("5 hundred 99 rupees"))
    assert prov.seen, "provider.synth was never called"
    # dot prepended, but digits NOT expanded (normalize is off)
    assert prov.seen[-1] == ".5 hundred 99 rupees"


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


# -- progressive pass-through stitch (opt-in streaming path) ----------------


def _spy_passthrough(svc, monkeypatch) -> list:
    """Record (in the returned list) whether the progressive pass-through
    generator is actually iterated. Delegates to the real method, so behavior
    is unchanged -- only observability is added."""
    used: list[bool] = []
    _orig = svc._progressive_stitch_stream

    async def _spy(*a, **kw):
        used.append(True)
        async for chunk in _orig(*a, **kw):
            yield chunk

    monkeypatch.setattr(svc, "_progressive_stitch_stream", _spy)
    return used


def _enable_passthrough(monkeypatch, min_words: int = 3) -> None:
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    monkeypatch.setattr(settings, "predictive_stitch_stream_enabled", True)
    monkeypatch.setattr(settings, "enable_pass_through_stitch", True)
    monkeypatch.setattr(settings, "pass_through_stitch_min_words", min_words)


async def test_pass_through_streams_prefix_then_gap_then_suffix_and_repeats_hit(
    svc, fake_provider, monkeypatch
):
    """Pass-through ON with a qualifying cached prefix: the progressive path
    streams the assembled clip, synthesizes ONLY the gap, and stores on
    completion so a repeat HITs."""
    _enable_passthrough(monkeypatch)
    for w in ("hello there friend", "how are you today"):  # cached prefix + suffix
        await svc.create(_req_text(w))
    seed_calls = fake_provider.calls
    used = _spy_passthrough(svc, monkeypatch)

    headers, gen = await svc.stream(_req_text("hello there friend NAME how are you today"))
    assert headers["X-Cache"] == "MISS-STITCH"
    audio = b"".join([c async for c in gen])
    assert len(audio) > 0
    assert used == [True]                       # the progressive path was taken
    assert fake_provider.calls == seed_calls + 1  # only the one gap ("NAME") synthesized

    # store-on-completion -> repeat is an instant HIT, no further synth
    h2, _g2 = await svc.stream(_req_text("hello there friend NAME how are you today"))
    assert h2["X-Cache"] == "HIT"
    assert fake_provider.calls == seed_calls + 1


async def test_pass_through_prefix_too_short_falls_back_to_assemble(
    svc, fake_provider, monkeypatch
):
    """Cached prefix < pass_through_stitch_min_words -> the progressive path is
    NOT used; the assemble path handles it (still MISS-STITCH, gap synth'd)."""
    _enable_passthrough(monkeypatch, min_words=3)
    for w in ("hi there", "how are you today"):  # 2-word prefix < 3
        await svc.create(_req_text(w))
    seed_calls = fake_provider.calls
    used = _spy_passthrough(svc, monkeypatch)

    headers, gen = await svc.stream(_req_text("hi there NAME how are you today"))
    assert headers["X-Cache"] == "MISS-STITCH"
    audio = b"".join([c async for c in gen])
    assert len(audio) > 0
    assert used == []                            # progressive path skipped (prefix too short)
    assert fake_provider.calls == seed_calls + 1  # assemble path still synth'd only the gap


async def test_pass_through_non_pcm_format_falls_back_to_assemble(
    svc, fake_provider, monkeypatch
):
    """Requested format != pcm_s16le@16k -> progressive path skipped (per-chunk
    resample is unsafe); the assemble path converts the whole clip instead."""
    _enable_passthrough(monkeypatch)
    for w in ("hello there friend", "how are you today"):
        await svc.create(_req_text(w))
    used = _spy_passthrough(svc, monkeypatch)

    req = _req_text("hello there friend NAME how are you today")
    req.output_format = OutputFormat(container="raw", encoding="mulaw", sample_rate=8000)
    headers, gen = await svc.stream(req)
    assert headers["X-Cache"] == "MISS-STITCH"
    audio = b"".join([c async for c in gen])
    assert len(audio) > 0
    assert used == []  # non-PCM -> assemble path


async def test_pass_through_flag_off_is_assemble_path(svc, fake_provider, monkeypatch):
    """Default flag OFF -> the progressive path is never touched; the existing
    assemble-then-stream path is byte-for-byte unchanged (the no-op guarantee)."""
    monkeypatch.setattr(settings, "predictive_stitch_enabled", True)
    monkeypatch.setattr(settings, "predictive_stitch_stream_enabled", True)
    monkeypatch.setattr(settings, "enable_pass_through_stitch", False)  # test the OFF path explicitly (default is True)
    for w in ("hello there friend", "how are you today"):
        await svc.create(_req_text(w))
    seed_calls = fake_provider.calls
    used = _spy_passthrough(svc, monkeypatch)

    headers, gen = await svc.stream(_req_text("hello there friend NAME how are you today"))
    assert headers["X-Cache"] == "MISS-STITCH"
    audio = b"".join([c async for c in gen])
    assert len(audio) > 0
    assert used == []                            # flag off -> progressive never used
    assert fake_provider.calls == seed_calls + 1  # assemble path synth'd only the gap

    h2, _g2 = await svc.stream(_req_text("hello there friend NAME how are you today"))
    assert h2["X-Cache"] == "HIT"


async def test_pass_through_eligible_but_first_span_is_gap_falls_back(
    svc, fake_provider, monkeypatch
):
    """Gap at the very start (no cached prefix before it) -> no TTFB win to claim,
    so the progressive path is skipped and the assemble path runs."""
    _enable_passthrough(monkeypatch)
    await svc.create(_req_text("how are you today"))  # only a SUFFIX cached; name leads
    used = _spy_passthrough(svc, monkeypatch)

    headers, gen = await svc.stream(_req_text("NAME how are you today"))
    assert headers["X-Cache"] == "MISS-STITCH"
    audio = b"".join([c async for c in gen])
    assert len(audio) > 0
    assert used == []  # no cached prefix -> assemble path


class _SlowGapProvider:
    """Cartesia stand-in whose synth sleeps, recording each gap's finish time so
    a test can prove the cached prefix streamed BEFORE the gap synth completed."""
    name = "cartesia"
    native_encoding = "pcm_s16le"
    native_sample_rate = 16000

    def __init__(self):
        self._audio = b"\x01\x00" * 16000   # 1s -- a real clip is >> the 30ms xfade window
        self.finish_times: dict[str, float] = {}

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        await asyncio.sleep(0.15)  # simulate synth latency
        self.finish_times[text] = time.monotonic()
        return AudioResult(self._audio, "raw", "pcm_s16le", 16000)


async def test_pass_through_emits_prefix_before_gap_synthesizes(tmp_storage, monkeypatch):
    """The feature's whole purpose: the cached prefix must reach the caller
    BEFORE the gap synth completes (TTFB ~0, not gap-synth-bound). A slow gap
    synth proves the ordering -- if the path ever regressed to assemble-style
    (wait for the gap first), the first chunk would arrive AFTER the gap finished."""
    _enable_passthrough(monkeypatch)
    prov = _SlowGapProvider()
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: prov if name == "cartesia" else None)

    await svc.create(_req_text("hello there friend"))   # cached prefix
    await svc.create(_req_text("how are you today"))    # cached suffix

    headers, gen = await svc.stream(_req_text("hello there friend NAME how are you today"))
    assert headers["X-Cache"] == "MISS-STITCH"
    first_chunk_at = None
    async for chunk in gen:
        if first_chunk_at is None:
            first_chunk_at = time.monotonic()
    assert first_chunk_at is not None
    assert "NAME" in prov.finish_times                       # the gap was synthesized
    assert first_chunk_at < prov.finish_times["NAME"], (     # prefix streamed FIRST
        first_chunk_at - prov.finish_times["NAME"]
    )


class _GapFailsProvider:
    """Cartesia stand-in that FAILS the gap synth (raises) but succeeds for the
    seeded prefix/suffix, to verify a mid-stream gap failure truncates cleanly."""
    name = "cartesia"
    native_encoding = "pcm_s16le"
    native_sample_rate = 16000

    def __init__(self):
        self._audio = b"\x01\x00" * 16000   # 1s -- a real clip is >> the 30ms xfade window

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        if text.strip() == "NAME":
            raise RuntimeError("simulated provider failure on the gap")
        return AudioResult(self._audio, "raw", "pcm_s16le", 16000)


async def test_pass_through_truncates_cleanly_on_gap_synth_error(tmp_storage, monkeypatch):
    """A gap-synth failure mid-stream (after headers + prefix are committed) must
    NOT raise out of the generator -- it can't become a clean 500 once audio
    headers are sent. It truncates (prefix delivered, stream ends), the failed
    synth is not credited, and nothing is cached."""
    _enable_passthrough(monkeypatch)
    prov = _GapFailsProvider()
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: prov if name == "cartesia" else None)

    await svc.create(_req_text("hello there friend"))
    await svc.create(_req_text("how are you today"))
    synth_before = (await meta.metrics_summary())["synth_calls"]  # 2, from the seeds

    headers, gen = await svc.stream(_req_text("hello there friend NAME how are you today"))
    assert headers["X-Cache"] == "MISS-STITCH"
    chunks = []
    async for chunk in gen:          # must NOT raise -- failure truncates
        chunks.append(chunk)
    audio = b"".join(chunks)
    assert len(audio) > 0            # the cached prefix was still delivered

    # the failed gap synth is NOT credited (synth_calls unchanged from the seeds)
    synth_after = (await meta.metrics_summary())["synth_calls"]
    assert synth_after == synth_before

    cached, _rec, _p, _m, _k = await svc.check(_req_text("hello there friend NAME how are you today"))
    assert cached is False           # truncated stream -> nothing cached for reuse


async def test_pass_through_synthesizes_multiple_gaps_concurrently(tmp_storage, monkeypatch):
    """Multiple gaps synthesize CONCURRENTLY (total ~= max gap, not the sum).
    Two gaps at ~0.15s each: sequential would be ~0.30s+, concurrent ~0.15-0.20s.
    Proves the gaps overlap, not run one-after-the-other."""
    _enable_passthrough(monkeypatch)
    prov = _SlowGapProvider()
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: prov if name == "cartesia" else None)

    await svc.create(_req_text("hello there friend"))   # cached prefix (3 words)
    await svc.create(_req_text("mid"))                  # cached middle (1 word)
    # spans: [0:3] cached, [3:4] gap "NAME", [4:5] cached "mid", [5:6] gap "NAME2"

    t0 = time.monotonic()
    headers, gen = await svc.stream(_req_text("hello there friend NAME mid NAME2"))
    assert headers["X-Cache"] == "MISS-STITCH"
    audio = b"".join([c async for c in gen])
    elapsed = time.monotonic() - t0

    assert len(audio) > 0
    assert "NAME" in prov.finish_times and "NAME2" in prov.finish_times  # both gaps ran
    # concurrent (<= ~0.2s), NOT sequential (>= ~0.3s). 0.27s splits the two with margin.
    assert elapsed < 0.27, f"gaps were sequential, not concurrent (elapsed={elapsed})"



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

