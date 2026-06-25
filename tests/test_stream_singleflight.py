"""Single-flight on /tts/stream: concurrent identical streaming MISSes share ONE
producer (one ``stream_synth`` call, one store) and the coalesced callers get the
full clip as a HIT.

The default FakeProvider returns instantly, so to GUARANTEE two requests overlap
(this is a race-sensitive property) the service-level test uses a latch-bearing
subclass defined here. The endpoint-level test uses the real ``app_client`` +
TestClient with the injected FakeProvider, driven concurrently via threads.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.cache.service import CacheService
from app.core.config import settings
from app.providers.base import AudioResult, BaseTTSProvider
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _pcm_req(text: str = "the quick brown fox") -> TTSRequest:
    """Native-format live-path request: pcm_s16le@16k (== FakeProvider native)."""
    return TTSRequest(
        model_id="cartesia:sonic-3.5",
        transcript=text,
        voice=CartesiaVoice(id="v1"),
        language="en",
        output_format=OutputFormat(
            container="raw", encoding="pcm_s16le", sample_rate=16000
        ),
    )


class LatchFakeProvider(BaseTTSProvider):
    """FakeProvider whose ``stream_synth`` BLOCKS on an asyncio.Event until a 2nd
    caller is observed, guaranteeing two concurrent ``svc.stream`` calls overlap
    (so the 2nd takes the coalesced path rather than running its own synth).

    ``calls_before_release`` counts stream_synth invocations; when the 2nd
    arrives we release the latch so the producer can finish and the coalesced
    waiter resolves.
    """

    name = "cartesia"

    def __init__(self, audio: bytes | None = None):
        self._audio = audio or (b"\x01\x00" * 400)
        self.stream_calls = 0
        self.calls = 0
        self._release = asyncio.Event()
        self._call_count = 0

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        self.calls += 1
        return AudioResult(
            audio=self._audio, container="raw",
            encoding="pcm_s16le", sample_rate=16000,
        )

    async def stream_synth(self, *, text, voice_id, model, language, params):
        self.stream_calls += 1
        self._call_count += 1
        if self._call_count < 2:
            # First (producer) caller: park until a 2nd caller shows up so the
            # second ``svc.stream`` definitely observes the in-flight future.
            try:
                await asyncio.wait_for(self._release.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        piece = max(1, len(self._audio) // 4)
        for i in range(0, len(self._audio), piece):
            yield self._audio[i : i + piece]


@pytest.fixture
async def svc(tmp_storage):
    """CacheService with the latch fake provider, started (so write-behind/stop
    lifecycle is exercised). Uses the autouse synchronous-metrics setting."""
    fake = LatchFakeProvider()
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: fake if name == "cartesia" else None)
    await svc.start()
    svc._fake = fake  # type: ignore[attr-defined]
    yield svc, fake
    await svc.stop()


# -- CacheService.stream: two concurrent MISSes for the same key ---------------


async def _run_concurrent_drains(svc_obj, fake, n: int) -> list[tuple[str, bytes]]:
    """Deterministically overlap ``n`` concurrent ``svc.stream(req)`` calls.

    The producer's stream_synth parks on ``fake._release``; we drive the first
    drain until the producer has ENTERED stream_synth (``_call_count >= 1``,
    which can only happen after its future is registered in ``_inflight``), and
    only THEN start the remaining drains — so they are guaranteed to observe the
    in-flight future and coalesce rather than becoming producers themselves.
    """
    req = _pcm_req()
    results: list[tuple[str, bytes]] = [None] * n  # type: ignore[list-item]
    pending: set[asyncio.Task] = set()

    async def drain(idx: int) -> None:
        headers, gen = await svc_obj.stream(req)
        chunks = [c async for c in gen]
        results[idx] = (headers["X-Cache"], b"".join(chunks))

    # Start the producer first.
    t0 = asyncio.create_task(drain(0))
    pending.add(t0)
    # Drive the loop until the producer has parked inside stream_synth (future
    # registered). Bounded by the provider's own wait_for timeout as a safety net.
    for _ in range(500):
        if fake._call_count >= 1:
            break
        await asyncio.sleep(0.005)
    # Now the in-flight future is registered -> start the coalesced callers.
    for i in range(1, n):
        pending.add(asyncio.create_task(drain(i)))
    # Let the coalesced callers resolve onto the future, then release the producer.
    for _ in range(200):
        await asyncio.sleep(0.005)
        if req_key_registered(svc_obj, req):
            break
    fake._release.set()
    await asyncio.gather(*pending)
    return results


def req_key_registered(svc_obj, req) -> bool:
    """True while the producer's future is still pending in ``_inflight``."""
    from app.cache.key import canonical_params, hash_key, parse_model_id

    provider, model = parse_model_id(req.model_id)
    key = hash_key(
        text=req.transcript, provider=provider, voice_id=req.voice.id, model=model,
        language=req.language, params_canonical=canonical_params(provider, req.params),
    )
    fut = svc_obj._inflight.get(key)
    return fut is not None and not fut.done()


async def test_two_concurrent_stream_misses_share_one_synth(svc):
    svc_obj, fake = svc
    res = await _run_concurrent_drains(svc_obj, fake, n=2)
    (s1, a1), (s2, a2) = res

    assert fake.stream_calls == 1                      # ONE synth, not two
    assert {s1, s2} == {"MISS", "HIT"}                # one MISS + one coalesced HIT
    assert a1 == a2 == fake._audio                     # both got the full clip
    # Stored exactly once -> a 3rd request is a HIT with no further synth.
    h3, g3 = await svc_obj.stream(_pcm_req())
    assert h3["X-Cache"] == "HIT"
    _ = [c async for c in g3]
    assert fake.stream_calls == 1


async def test_three_concurrent_stream_misses_share_one_synth(svc):
    """N>2 concurrent identical stream MISSes still collapse to one synth."""
    svc_obj, fake = svc
    res = await _run_concurrent_drains(svc_obj, fake, n=3)
    statuses = [s for s, _ in res]
    audios = [a for _, a in res]
    assert fake.stream_calls == 1
    assert statuses.count("MISS") == 1
    assert statuses.count("HIT") == 2
    assert all(a == fake._audio for a in audios)


async def test_stream_single_flight_stores_once(svc):
    """Coalesced streaming still stores the clip exactly once (the producer)."""
    svc_obj, fake = svc
    await _run_concurrent_drains(svc_obj, fake, n=2)

    cached, record, *_ = await svc_obj.check(_pcm_req())
    assert cached is True
    assert record.size_bytes == len(fake._audio)  # one store, full clip


async def test_coalesced_caller_survives_producer_failure(svc, monkeypatch):
    """If the producer's stream fails, the coalesced waiter falls back to its
    OWN synth rather than 502-ing on the peer's failure (so total synths == 2)."""
    svc_obj, _fake = svc
    req = _pcm_req()

    # Second FakeProvider that errors on its FIRST stream_synth but succeeds
    # thereafter, so the coalesced waiter's fallback synth works.
    class FailingThenOK(BaseTTSProvider):
        name = "cartesia"

        def __init__(self):
            self.stream_calls = 0
            self._audio = b"\x02\x00" * 300

        async def synth(self, *, text, voice_id, model, language, params):
            return AudioResult(
                audio=self._audio, container="raw",
                encoding="pcm_s16le", sample_rate=16000,
            )

        async def stream_synth(self, *, text, voice_id, model, language, params):
            self.stream_calls += 1
            if self.stream_calls == 1:
                raise RuntimeError("producer blew up mid-stream")
            yield self._audio

    fail_fake = FailingThenOK()
    # Swap the provider the CacheService resolves for "cartesia".
    svc_obj._get_provider = lambda name: fail_fake if name == "cartesia" else None  # type: ignore[assignment]
    # Clear any in-flight + cached state for this key from the fixture warmup.
    svc_obj._inflight.clear()

    async def drain() -> tuple[str, bytes]:
        headers, gen = await svc_obj.stream(req)
        try:
            chunks = [c async for c in gen]
            return headers["X-Cache"], b"".join(chunks)
        except Exception as e:  # the producer surfaces its own failure
            return "ERR", str(e).encode()

    job = asyncio.gather(drain(), drain())
    (s1, a1), (s2, a2) = await job

    # Producer failed on its own stream (its caller sees ERR / raised); the
    # coalesced waiter fell back and synthesized itself -> 2 stream_synth calls.
    assert fail_fake.stream_calls == 2
    # One of them failed (ERR surfaced to the producer's consumer); the other
    # succeeded and got the full fallback audio.
    assert ("ERR" in (s1, s2)) and (fail_fake._audio in (a1, a2))


# -- /tts/stream endpoint: 2 concurrent via TestClient, one MISS + one HIT ------


def test_endpoint_two_concurrent_streams_single_flight(app_client, monkeypatch):
    """Two concurrent /tts/stream POSTs for the same uncached key via the real
    FastAPI app: the X-Cache headers show one MISS + one coalesced HIT and the
    provider's stream_synth runs exactly once.

    Determinism: the orchestrating thread (not a coalesced caller) releases the
    producer. The producer parks inside stream_synth AFTER its in-flight future
    is registered, so the second request is guaranteed to observe it and
    coalesce rather than becoming a second producer. Stitch is disabled here so
    the live single-flight path is the one exercised (stitch is checked before
    single-flight in stream() and has its own dedicated tests).
    """
    import time

    # Isolate single-flight: keep the stream request on the live path (no stitch).
    monkeypatch.setattr(settings, "predictive_stitch_enabled", False)
    monkeypatch.setattr(settings, "predictive_stitch_stream_enabled", False)

    class ThreadLatchFake(BaseTTSProvider):
        name = "cartesia"

        def __init__(self):
            self._audio = b"\x03\x00" * 400
            self.stream_calls = 0
            self.calls = 0
            self._arrived = threading.Event()
            self._release = threading.Event()
            self._lock = threading.Lock()

        async def synth(self, *, text, voice_id, model, language, params):
            self.calls += 1
            return AudioResult(
                audio=self._audio, container="raw",
                encoding="pcm_s16le", sample_rate=16000,
            )

        async def stream_synth(self, *, text, voice_id, model, language, params):
            with self._lock:
                self.stream_calls += 1
            # Signal that the producer has entered stream_synth — its in-flight
            # future was registered upstream in stream() before this generator
            # ran, so a concurrent caller will observe and coalesce onto it.
            self._arrived.set()
            # Park until the orchestrator releases us (it does so only after the
            # second request has had time to coalesce onto the future).
            self._release.wait(timeout=5.0)
            piece = max(1, len(self._audio) // 4)
            for i in range(0, len(self._audio), piece):
                yield self._audio[i : i + piece]

    fake = ThreadLatchFake()
    app_client.app.state.registry._providers["cartesia"] = fake

    req = {
        "model_id": "cartesia:sonic-3.5",
        "transcript": "concurrent stream singleflight endpoint",
        "voice": {"id": "v1"},
        "language": "en",
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
    }

    results: dict[int, tuple[int, str | None, bytes]] = {}
    errors: dict[int, str] = {}

    def call(tag: int):
        try:
            r = app_client.post("/tts/stream", json=req)
            results[tag] = (r.status_code, r.headers.get("X-Cache"), r.content)
        except Exception as e:  # pragma: no cover - surfaced via assertion below
            errors[tag] = repr(e)

    t1 = threading.Thread(target=call, args=(1,))
    t2 = threading.Thread(target=call, args=(2,))
    t1.start()
    # Wait until the producer has registered its in-flight future (it signals
    # _arrived from inside stream_synth, which runs only after stream() returned
    # and registered the future). This is the anchor that makes t2 coalesce.
    assert fake._arrived.wait(timeout=3.0), "producer never entered stream_synth"
    time.sleep(0.05)  # let the future sit in _inflight
    t2.start()        # t2 observes the in-flight future -> coalesces (HIT)
    time.sleep(0.15)  # let t2 reach the await on the future
    fake._release.set()  # release the producer -> it streams, completes, resolves t2
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"requests errored: {errors}"
    assert len(results) == 2
    statuses = {v[1] for v in results.values()}
    assert statuses == {"MISS", "HIT"}, f"expected one MISS + one HIT, got {statuses}"
    # Both responses carry the full clip.
    for _code, _h, body in results.values():
        assert body == fake._audio
    # Exactly one provider stream_synth ran (single-flight).
    assert fake.stream_calls == 1
