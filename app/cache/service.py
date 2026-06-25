"""Cache service — lookup / synth / store-native / convert-on-serve + metrics.

The cache key is format-agnostic (text + provider + voice + model + language +
params). Audio is stored once in the provider's *native* format and converted to
the caller's requested ``output_format`` on serve, so a single entry serves
every format — the one-shot μ-law path and the streaming PCM path share it.

Read path: cache check → HIT loads native, converts to requested, returns
(+ records a hit) → MISS synthesizes native, write-throughs native, converts,
returns (+ records a miss). Streaming MISS forwards native chunks live (when
the requested format == native) and stores native on clean completion.
Admin: check / create / delete / clear. Metrics flow into a daily rollup; the
cache snapshot is served from incrementally-maintained totals.
"""

from __future__ import annotations

import asyncio

import numpy as np
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Callable

from app.audio.format import convert_audio
from app.cache.metrics import WriteBehindMetrics
from app.cache.resilience import get_gate
from app.cache.key import (
    canonical_params,
    hash_key,
    normalize_text,
    parse_model_id,
)
from app.cache.segment import segment
from app.core.config import settings
from app.core.logging import logger
from app.providers.base import AudioResult, BaseTTSProvider, ProviderError
from app.providers.registry import ProviderNotConfigured
from app.schemas.tts import TTSRequest
from app.storage.base import CacheRecord

# Sentinel: caller has NOT pre-fetched the existing record (vs. None = checked,
# genuinely absent). Avoids a redundant get() on every store when the caller
# already knows.
_UNCHECKED = object()

# Chunk size used when streaming a complete blob back to the caller (HIT, or a
# one-shot miss). Large enough to amortize per-iteration overhead, small enough
# to start flowing promptly.
_STREAM_CHUNK = 16 * 1024


async def _chunked(data: bytes, size: int = _STREAM_CHUNK) -> AsyncGenerator[bytes, None]:
    """Yield ``data`` in fixed-size byte chunks for an HTTP streaming response."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


# --- stitch assembly DSP (numpy) ------------------------------------------
# Assembles native pcm_s16le@16k mono sub-phrase clips into one continuous clip.
# Choppiness in stitched TTS comes from three cheap-to-fix things — not pitch:
#   1. clicks/pops  -> DC-offset removal + splice at a zero crossing
#   2. doubled silence -> edge silence-trim (with a guard so onsets survive)
#   3. loudness mismatch -> per-clip RMS normalization to a common target
# plus an equal-power (constant-power) crossfade so the seam doesn't dip.
# Pure numpy: no heavy voice deps, and this is also the audioop->Py3.13 path.

_SR = 16_000              # native sample rate (pcm_s16le@16k)
_XFADE_MS = 15.0          # equal-power crossfade overlap (~15ms; 10-25ms is the sweet spot)
_TARGET_RMS_DB = -20.0    # per-clip loudness target (speech sits ~-23..-18 dBFS RMS)
_RMS_FLOOR_DB = -55.0     # below this a clip is near-silent (breath/gap): don't amplify
_SIL_THRESH = 0.012       # ~1% full-scale; windowed RMS below this = silence
_SIL_GUARD_MS = 6.0       # guard kept at each trimmed edge so word onsets/offsets survive
_ZC_SEARCH_MS = 4.0       # window scanned for a zero crossing to anchor the splice


def _to_float(b: bytes) -> np.ndarray:
    """int16 LE pcm bytes -> float32 in [-1, 1]."""
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def _to_int16(x: np.ndarray) -> bytes:
    """float32 [-1, 1] -> int16 LE pcm bytes."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _rms_normalize(x: np.ndarray) -> np.ndarray:
    """Scale so the clip's RMS hits the target loudness; leave near-silent clips."""
    rms = float(np.sqrt(np.mean(x * x))) + 1e-12
    db = 20.0 * np.log10(rms)
    if db < _RMS_FLOOR_DB:
        return x
    return x * (10.0 ** ((_TARGET_RMS_DB - db) / 20.0))


def _trim_edges(x: np.ndarray, head: bool, tail: bool) -> np.ndarray:
    """Trim leading/trailing near-silent samples (windowed RMS), keeping a guard."""
    n = len(x)
    if n == 0:
        return x
    win = max(1, int(_SR * 1.0 / 1000))  # 1ms RMS window
    energy = np.convolve(x * x, np.ones(win, dtype=np.float32) / win, mode="same")
    loud = energy > _SIL_THRESH * _SIL_THRESH
    if not loud.any():
        return x  # all-silent (e.g. test fixtures): leave untouched
    first = int(np.argmax(loud))
    last = n - int(np.argmax(loud[::-1]))
    guard = int(_SR * _SIL_GUARD_MS / 1000)
    start = max(0, first - guard) if head else 0
    end = min(n, last + guard) if tail else n
    if start >= end:
        return x
    return x[start:end]


def _snap_zero(x: np.ndarray, side: str) -> int:
    """Index near the head/tail edge where the signal crosses zero (minimizes the
    step discontinuity at the splice). Falls back to the raw edge if none nearby."""
    span = min(int(_SR * _ZC_SEARCH_MS / 1000), len(x))
    if span < 2:
        return 0 if side == "head" else len(x)
    if side == "head":
        seg = x[:span]
        zc = np.where(np.diff(np.signbit(seg)))[0]
        return int(zc[0]) if zc.size else 0
    seg = x[-span:]
    zc = np.where(np.diff(np.signbit(seg)))[0]
    return (len(x) - span + int(zc[-1] + 1)) if zc.size else len(x)


def _equal_power_xfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap a's tail with b's head via a constant-power (cos/sin) crossfade.
    cos^2+sin^2=1 keeps total power constant through the fade, avoiding the -3dB
    midpoint dip a linear fade produces on uncorrelated speech."""
    n = min(int(_SR * _XFADE_MS / 1000), len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)  # 0 -> pi/2
    blended = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], blended, b[n:]])


def _stitch_clips(pieces: list[bytes]) -> bytes:
    """Assemble native pcm_s16le@16k clips into one continuous clip.

    Per clip: DC-remove -> silence-trim -> RMS-normalize; then splice each pair
    at zero crossings with an equal-power crossfade. Returns native int16 bytes.
    """
    if not pieces:
        return b""
    clips: list[np.ndarray] = []
    for p in pieces:
        x = _to_float(p)
        x = _trim_edges(x, head=True, tail=True)  # trim silence on the raw signal first
        x = x - float(np.mean(x))                  # DC removal over the voiced part
        x = _rms_normalize(x)
        clips.append(x)
    if len(clips) == 1:
        return _to_int16(clips[0])
    out = clips[0]
    for nxt in clips[1:]:
        out = out[: _snap_zero(out, "tail")]
        nxt = nxt[_snap_zero(nxt, "head"):]
        out = _equal_power_xfade(out, nxt)
    return _to_int16(out)


def _same_format(encoding_a: str, rate_a: int, encoding_b: str, rate_b: int) -> bool:
    return encoding_a.lower() == encoding_b.lower() and rate_a == rate_b


class CacheService:
    def __init__(
        self,
        metadata,
        blobs,
        get_provider: Callable[[str], BaseTTSProvider | None],
    ):
        self._metadata = metadata
        self._blobs = blobs
        self._get_provider = get_provider
        # Write-behind wrapper for stats-only writes (HIT touch + metrics) so a HIT
        # returns audio without awaiting a SQLite commit. Falls back to the raw
        # store (synchronous) when disabled. Correctness-critical writes
        # (put/put_with_totals/delete/adjust_totals) stay on self._metadata.
        self._metrics = (
            WriteBehindMetrics(
                metadata,
                settings.metrics_flush_interval_ms / 1000.0,
                settings.metrics_flush_batch_size,
            )
            if settings.metrics_write_behind_enabled
            else metadata
        )
        # Ephemeral session counters (reset on restart); durable metrics are in the DB.
        self._hits = 0
        self._misses = 0
        # Per-key in-flight synth futures for single-flight: N concurrent
        # identical MISSes share ONE synth + ONE store. Loop-thread-only (no lock).
        self._inflight: dict[str, asyncio.Future] = {}
        # Optional predictive-warmer observer (set via attach_tracker). Never
        # blocks or breaks reads; observation is fast and warming is async.
        self._tracker = None

    def attach_tracker(self, tracker) -> None:
        self._tracker = tracker

    async def start(self) -> None:
        """Start background tasks (write-behind metrics flusher)."""
        if hasattr(self._metrics, "start"):
            self._metrics.start()

    async def stop(self) -> None:
        """Flush + stop background tasks (graceful shutdown loses no metrics)."""
        if hasattr(self._metrics, "stop"):
            await self._metrics.stop()

    async def _observe(self, *, text, provider, voice_id, model, language, params) -> None:
        if self._tracker is None:
            return
        try:
            await self._tracker.observe(
                text=text,
                provider=provider,
                voice_id=voice_id,
                model=model,
                language=language,
                params=params,
            )
        except Exception as e:
            logger.warning(f"cache observe failed: {e}")

    # -- internals -----------------------------------------------------------

    def _resolve(self, req: TTSRequest):
        provider, model = parse_model_id(req.model_id)
        of = req.output_format
        params_canon = canonical_params(provider, req.params)
        key = hash_key(
            text=req.transcript,
            provider=provider,
            voice_id=req.voice.id,
            model=model,
            language=req.language,
            params_canonical=params_canon,
        )
        return provider, model, of, params_canon, key

    @staticmethod
    def _expired(record: CacheRecord) -> bool:
        return bool(
            record.ttl_expires_at
            and datetime.fromisoformat(record.ttl_expires_at) <= datetime.now(timezone.utc)
        )

    async def _synthesize(self, req: TTSRequest, provider: str, model: str) -> AudioResult:
        """Synthesize via the routed provider; return its NATIVE-format audio.

        Runs under the provider's resilience gate (bulkhead + rate limit) so a
        slow/hung provider is capped and can't starve the others.
        """
        instance = self._get_provider(provider)
        if instance is None:
            raise ProviderNotConfigured(provider)
        gate = get_gate(provider)
        async with gate:
            return await instance.synth(
                text=req.transcript,
                voice_id=req.voice.id,
                model=model,
                language=req.language,
                params=req.params,
            )

    async def _store(
        self,
        key: str,
        req: TTSRequest,
        provider: str,
        model: str,
        params_canon: str,
        audio: bytes,
        encoding: str,
        sample_rate: int,
        existing=_UNCHECKED,
        replace: bool = False,
    ) -> None:
        """Store ``audio`` (already in ``encoding``/``sample_rate``) as native.

        Fresh misses (no existing row, ``replace=False``) use INSERT OR IGNORE so
        concurrent identical misses can't double-count ``provider_totals``.
        Existing rows (expired refresh) and explicit overrides (``replace=True``)
        REPLACE the row and adjust totals by the size delta.
        """
        if existing is _UNCHECKED:
            existing = await self._metadata.get(key)
        old_size = existing.size_bytes if existing else 0
        now = datetime.now(timezone.utc)
        storage_path = await self._blobs.put(key, audio)
        ttl = (
            (now + timedelta(seconds=settings.ttl_seconds)).isoformat()
            if settings.ttl_seconds
            else None
        )
        record = CacheRecord(
            key=key,
            provider=provider,
            voice_id=req.voice.id,
            model=model,
            language=req.language,
            params=params_canon,
            text=req.transcript,
            container="raw",
            encoding=encoding,
            sample_rate=sample_rate,
            size_bytes=len(audio),
            storage_path=storage_path,
            hit_count=0,
            created_at=now.isoformat(),
            last_accessed_at=now.isoformat(),
            ttl_expires_at=ttl,
        )
        if existing is None and not replace:
            # Fresh miss: INSERT OR IGNORE — race-free totals under concurrent
            # identical misses (only the first store inserts + bumps totals).
            await self._metadata.put_with_totals(record)
        else:
            # Existing row (refresh) or explicit override: REPLACE + size delta.
            await self._metadata.put(record)
            await self._metadata.adjust_totals(
                provider, 0 if existing else 1, len(audio) - old_size
            )

    async def _convert_audio(
        self, data: bytes, *, native_encoding: str, native_rate: int,
        out_encoding: str, out_rate: int,
    ) -> bytes:
        """convert_audio off the event loop.

        audioop resample/μ-law conversion is CPU-bound and runs on every serve
        (HIT and MISS); inline it would block the loop (and starve the Cartesia
        socket pumps) under concurrent load, so dispatch it to the worker pool.
        """
        return await asyncio.to_thread(
            convert_audio, data,
            native_encoding=native_encoding, native_rate=native_rate,
            out_encoding=out_encoding, out_rate=out_rate,
        )

    # -- read path -----------------------------------------------------------

    async def stitch(self, req: TTSRequest, provider: str, model: str, params_canon: str):
        """Serve a full-text MISS from cached sub-phrases where possible.

        Binary-searches the longest cached prefix/suffix (monotonic under the
        substring-closed cache), recurses on the middle, synthesizes only the
        gaps, then cross-fades the pieces together. Returns native
        ``pcm_s16le@16k`` audio, or ``None`` if stitching isn't worthwhile
        (disabled, too short, or cached coverage below the gate) so the caller
        synthesizes the whole phrase.
        """
        if not settings.predictive_stitch_enabled:
            return None
        words = normalize_text(req.transcript).split()
        if len(words) < 2:
            return None

        async def is_cached(lo: int, hi: int) -> bool:
            skey = hash_key(
                text=" ".join(words[lo:hi]), provider=provider, voice_id=req.voice.id,
                model=model, language=req.language, params_canonical=params_canon,
            )
            rec = await self._metadata.get(skey)
            return bool(rec and not self._expired(rec))

        spans = await segment(len(words), is_cached)
        cached_words = sum(hi - lo for lo, hi, c in spans if c)
        if cached_words == 0 or cached_words / len(words) < settings.predictive_stitch_min_coverage:
            return None  # not enough cached -> let the caller synth the whole phrase

        async def span_audio(lo: int, hi: int, c: bool) -> bytes:
            sub_text = " ".join(words[lo:hi])
            if c:
                skey = hash_key(
                    text=sub_text, provider=provider, voice_id=req.voice.id,
                    model=model, language=req.language, params_canonical=params_canon,
                )
                rec = await self._metadata.get(skey)
                if rec and not self._expired(rec):
                    return await self._blobs.get(rec.storage_path)
                # evicted between probe and fetch -> fall through to synth
            sub_req = TTSRequest(
                model_id=req.model_id, transcript=sub_text, voice=req.voice,
                language=req.language, output_format=req.output_format, params=req.params,
            )
            native = await self._synthesize(sub_req, provider, model)
            return native.audio

        pieces = await asyncio.gather(*(span_audio(lo, hi, c) for (lo, hi, c) in spans))
        logger.info(
            f"STITCH key=… provider={provider} words={len(words)} "
            f"spans={len(spans)} coverage={cached_words}/{len(words)}"
        )
        return await asyncio.to_thread(_stitch_clips, pieces)

    async def _produce(
        self, req: TTSRequest, provider: str, model: str, params_canon: str,
        key: str, record,
    ) -> tuple[bytes, str, int, str]:
        """Synth + store for a MISS. Returns (native, encoding, rate, status)
        where status is 'MISS', 'MISS-STITCH', or 'HIT' (if the cache was filled
        between the outer lookup and here — e.g. by the warmer)."""
        rec = await self._metadata.get(key)
        if rec and not self._expired(rec):
            return await self._blobs.get(rec.storage_path), rec.encoding, rec.sample_rate, "HIT"
        stitched = await self.stitch(req, provider, model, params_canon)
        if stitched is not None:
            if settings.enable_write_through:
                await self._store(
                    key, req, provider, model, params_canon,
                    stitched, "pcm_s16le", 16000, existing=record,
                )
            return stitched, "pcm_s16le", 16000, "MISS-STITCH"
        native = await self._synthesize(req, provider, model)
        if settings.enable_write_through:
            await self._store(
                key, req, provider, model, params_canon,
                native.audio, native.encoding, native.sample_rate, existing=record,
            )
        return native.audio, native.encoding, native.sample_rate, "MISS"

    async def _produce_native(
        self, req: TTSRequest, provider: str, model: str, params_canon: str,
        key: str, record,
    ) -> tuple[bytes, str, int, str, bool]:
        """Single-flight: produce + store native audio for ``key`` exactly once.

        Concurrent MISSes for the same key share one synth + one store (the
        producer's future). Returns (native, encoding, rate, status, produced):
        ``produced`` is True for the caller that ran the synth (records a miss +
        synth_call) and False for a caller that coalesced (records a hit — served
        without its own Cartesia call).
        """
        fut = self._inflight.get(key)
        if fut is not None:
            native, enc, rate, status = await fut
            return native, enc, rate, status, False  # coalesced — no own synth
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._inflight[key] = fut
        try:
            result = await self._produce(req, provider, model, params_canon, key, record)
            if not fut.done():
                fut.set_result(result)
            native, enc, rate, status = result
            # `produced` = a synth actually ran. _produce can short-circuit to
            # status='HIT' (the warmer filled the key between the outer MISS
            # lookup and the inner re-fetch) — that served a cache read, not a
            # synth, so it must count as a HIT, not a miss/synth_call.
            produced = status in ("MISS", "MISS-STITCH")
            return native, enc, rate, status, produced
        except BaseException as e:
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            if self._inflight.get(key) is fut:
                self._inflight.pop(key, None)

    async def get_or_synthesize(self, req: TTSRequest) -> tuple[bytes, dict]:
        provider, model, of, params_canon, key = self._resolve(req)
        await self._observe(
            text=req.transcript, provider=provider, voice_id=req.voice.id,
            model=model, language=req.language, params=req.params,
        )

        record = await self._metadata.get(key)
        if record and not self._expired(record):
            native = await self._blobs.get(record.storage_path)
            audio = await self._convert_audio(
                native,
                native_encoding=record.encoding,
                native_rate=record.sample_rate,
                out_encoding=of.encoding,
                out_rate=of.sample_rate,
            )
            await self._metrics.touch_and_record(
                key, {"requests": 1, "hits": 1, "bytes_served": len(audio)}
            )
            self._hits += 1
            logger.info(f"CACHE HIT  key={key[:12]}… provider={provider}")
            return audio, {"X-Cache": "HIT", "X-Cache-Key": key}

        logger.info(f"CACHE MISS key={key[:12]}… provider={provider} — synthesizing")

        # Single-flight: N concurrent identical misses share one synth + one store.
        native, nenc, nrate, status, produced = await self._produce_native(
            req, provider, model, params_canon, key, record
        )
        audio = await self._convert_audio(
            native,
            native_encoding=nenc,
            native_rate=nrate,
            out_encoding=of.encoding,
            out_rate=of.sample_rate,
        )
        if produced:
            await self._metrics.record_metrics(
                requests=1, misses=1, bytes_served=len(audio), synth_calls=1
            )
            self._misses += 1
            logger.info(f"CACHE {status} key={key[:12]}… provider={provider}")
        else:
            # Served from cache without our own synth: either coalesced onto an
            # in-flight producer, or _produce found the warmer had filled the key
            # (status='HIT'). Count as a HIT AND bump hit_count/last_accessed —
            # the normal HIT path does this via touch_and_record; the single-
            # flight path must too, or LRU recency goes stale under warming.
            await self._metrics.touch_and_record(
                key, {"requests": 1, "hits": 1, "bytes_served": len(audio)}
            )
            self._hits += 1
            status = "HIT"
            logger.info(f"CACHE HIT (coalesced) key={key[:12]}… provider={provider}")
        return audio, {"X-Cache": status, "X-Cache-Key": key}

    # -- streaming read path ------------------------------------------------

    async def stream(self, req: TTSRequest) -> tuple[dict, AsyncGenerator[bytes, None]]:
        """Resolve ``req`` and return (response headers, audio-chunk generator).

        HIT: native blob loaded, converted to the requested format if needed,
        then streamed in fixed-size chunks.
        MISS (requested == native): provider chunks forwarded live (low TTFB),
        native clip stored on clean completion.
        MISS (requested != native): can't resample per-chunk, so synthesize
        native fully, store native, convert, then chunk.
        """
        provider, model, of, params_canon, key = self._resolve(req)
        await self._observe(
            text=req.transcript, provider=provider, voice_id=req.voice.id,
            model=model, language=req.language, params=req.params,
        )

        record = await self._metadata.get(key)
        if record and not self._expired(record):
            native = await self._blobs.get(record.storage_path)
            if _same_format(of.encoding, of.sample_rate, record.encoding, record.sample_rate):
                served, chunks = native, _chunked(native)
            else:
                served = await self._convert_audio(
                    native,
                    native_encoding=record.encoding,
                    native_rate=record.sample_rate,
                    out_encoding=of.encoding,
                    out_rate=of.sample_rate,
                )
                chunks = _chunked(served)
            await self._metrics.touch_and_record(
                key, {"requests": 1, "hits": 1, "bytes_served": len(served)}
            )
            self._hits += 1
            logger.info(f"CACHE HIT  (stream) key={key[:12]}… provider={provider}")
            return {"X-Cache": "HIT", "X-Cache-Key": key}, chunks

        logger.info(
            f"CACHE MISS (stream) key={key[:12]}… provider={provider} — streaming synth"
        )

        # Predictive stitch: assemble from cached sub-phrases + synthesized gaps,
        # store the assembled clip, and stream it chunked (so future requests HIT).
        # Falls through to live streaming if not worthwhile. Unlike live streaming
        # this waits for the gaps to synth before first byte — but the assembled
        # clip is then cached, so repeats are instant HITs.
        if settings.predictive_stitch_stream_enabled:
            stitched = await self.stitch(req, provider, model, params_canon)
            if stitched is not None:
                if settings.enable_write_through:
                    await self._store(
                        key, req, provider, model, params_canon,
                        stitched, "pcm_s16le", 16000, existing=record,
                    )
                audio = await self._convert_audio(
                    stitched,
                    native_encoding="pcm_s16le",
                    native_rate=16000,
                    out_encoding=of.encoding,
                    out_rate=of.sample_rate,
                )
                await self._metrics.record_metrics(
                    requests=1, misses=1, bytes_served=len(audio), synth_calls=1
                )
                self._misses += 1
                logger.info(
                    f"CACHE MISS-STITCH (stream) key={key[:12]}… "
                    f"provider={provider} size={len(audio)}B"
                )
                return {"X-Cache": "MISS-STITCH", "X-Cache-Key": key}, _chunked(audio)

        instance = self._get_provider(provider)
        if instance is None:
            raise ProviderNotConfigured(provider)

        if _same_format(of.encoding, of.sample_rate, instance.native_encoding, instance.native_sample_rate):
            # Single-flight: if a synth is already in-flight for this key (bytes
            # or stream path), coalesce onto it — await the result and stream the
            # completed clip — instead of opening a 2nd provider stream. Else
            # become the producer: register the future BEFORE any await so a
            # concurrent request coalesces onto us (loop-thread-atomic).
            fut = self._inflight.get(key)
            if fut is not None and not fut.done():
                return await self._stream_coalesced(
                    req, key, fut, of, provider, instance, model, params_canon, record
                )
            fut = asyncio.get_running_loop().create_future()
            self._inflight[key] = fut
            return (
                {"X-Cache": "MISS", "X-Cache-Key": key},
                self._stream_and_store(
                    req, instance, key, provider, model, params_canon, record,
                    instance.native_encoding, instance.native_sample_rate, fut,
                ),
            )

        # Requested format differs from native: synth fully, store native, convert.
        native = await self._synthesize(req, provider, model)
        if settings.enable_write_through:
            await self._store(
                key, req, provider, model, params_canon,
                native.audio, native.encoding, native.sample_rate, existing=record,
            )
        audio = await self._convert_audio(
            native.audio,
            native_encoding=native.encoding,
            native_rate=native.sample_rate,
            out_encoding=of.encoding,
            out_rate=of.sample_rate,
        )
        await self._metrics.record_metrics(
            requests=1, misses=1, bytes_served=len(audio), synth_calls=1
        )
        self._misses += 1
        return {"X-Cache": "MISS", "X-Cache-Key": key}, _chunked(audio)

    async def _stream_and_store(
        self,
        req: TTSRequest,
        instance: BaseTTSProvider,
        key: str,
        provider: str,
        model: str,
        params_canon: str,
        record,
        native_encoding: str,
        native_sample_rate: int,
        fut: asyncio.Future,
    ) -> AsyncGenerator[bytes, None]:
        """Forward native provider chunks to the caller; store native on success.

        Partial audio is never cached: if the consumer stops early (client
        disconnect) or the provider errors, ``completed`` stays False and the
        accumulated bytes are discarded. The inner provider generator is closed
        explicitly so its socket is released promptly.
        """
        accumulated = bytearray()
        completed = False
        gate = get_gate(provider)
        gen = instance.stream_synth(
            text=req.transcript,
            voice_id=req.voice.id,
            model=model,
            language=req.language,
            params=req.params,
        )
        try:
            # Hold the gate for the whole stream — it IS an in-flight synth.
            async with gate:
                async for chunk in gen:
                    accumulated += chunk
                    yield chunk
                completed = True
        finally:
            await gen.aclose()
            if completed and accumulated:
                audio = bytes(accumulated)
                if settings.enable_write_through:
                    await self._store(
                        key, req, provider, model, params_canon,
                        audio, native_encoding, native_sample_rate, existing=record,
                    )
                await self._metrics.record_metrics(
                    requests=1, misses=1, bytes_served=len(audio), synth_calls=1
                )
                self._misses += 1
                if not fut.done():
                    fut.set_result((audio, native_encoding, native_sample_rate, "MISS"))
                logger.info(
                    f"CACHE STORE (stream) key={key[:12]}… provider={provider} size={len(audio)}B"
                )
            elif not fut.done():
                # Aborted (client disconnect) or provider error: partial audio is
                # discarded (never cached). Signal coalesced waiters to fall back
                # to their own synth rather than 502 on someone else's failure.
                fut.set_exception(ProviderError("stream synth failed or aborted"))
            if self._inflight.get(key) is fut:
                self._inflight.pop(key, None)

    async def _stream_coalesced(
        self, req, key, fut, of, provider, instance, model, params_canon, record,
    ) -> tuple[dict, AsyncGenerator[bytes, None]]:
        """Serve a streaming MISS by awaiting an in-flight producer for ``key``.

        The producer (bytes- or stream-path) is already synthesizing; we wait for
        its result, then stream the completed clip in chunks (a coalesced HIT). If
        the producer aborted/failed, fall through to our own live synth — by the
        time ``await fut`` raises, the producer has cleared ``_inflight``, so we're
        the only contender. (CancelledError is NOT caught: a cancelled waiter dies.)
        """
        try:
            native, enc, rate, _status = await fut
        except Exception:
            # Producer failed. Another waiter may already have become the new
            # producer — re-check _inflight and coalesce onto it instead of each
            # waiter spawning its own synth (bounds a producer failure to ONE
            # retry, not N).
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                return await self._stream_coalesced(
                    req, key, existing, of, provider, instance, model, params_canon, record
                )
            fut = asyncio.get_running_loop().create_future()
            self._inflight[key] = fut
            return (
                {"X-Cache": "MISS", "X-Cache-Key": key},
                self._stream_and_store(
                    req, instance, key, provider, model, params_canon, record,
                    instance.native_encoding, instance.native_sample_rate, fut,
                ),
            )
        served = (
            native
            if _same_format(of.encoding, of.sample_rate, enc, rate)
            else await self._convert_audio(
                native, native_encoding=enc, native_rate=rate,
                out_encoding=of.encoding, out_rate=of.sample_rate,
            )
        )
        await self._metrics.touch_and_record(
            key, {"requests": 1, "hits": 1, "bytes_served": len(served)}
        )
        self._hits += 1
        logger.info(f"CACHE HIT (stream-coalesced) key={key[:12]}… provider={provider}")
        return {"X-Cache": "HIT", "X-Cache-Key": key}, _chunked(served)

    # -- admin ops -----------------------------------------------------------

    async def check(self, req: TTSRequest):
        """Return (cached, record_or_none, provider, model, key). No synthesis."""
        provider, model, of, params_canon, key = self._resolve(req)
        record = await self._metadata.get(key)
        cached = bool(record and not self._expired(record))
        return cached, record, provider, model, key

    async def create(self, req: TTSRequest, audio_override: bytes | None = None):
        """Force create/override. Returns
        (key, status, source, size_bytes, provider, model, stored_encoding,
        stored_sample_rate).

        Synthesized audio is stored in native format; a base64 override is
        stored as-is in the requested output_format (the caller asserts the
        supplied audio matches it). The returned encoding/rate is what's
        ACTUALLY stored. Either way the format-agnostic key means the entry
        serves every requested format on read.
        """
        provider, model, of, params_canon, key = self._resolve(req)
        existing = await self._metadata.get(key)
        overridden = bool(existing and not self._expired(existing))

        if audio_override is not None:
            audio = audio_override
            store_encoding, store_rate = of.encoding, of.sample_rate
            source = "base64"
        else:
            native = await self._synthesize(req, provider, model)
            audio = native.audio
            store_encoding, store_rate = native.encoding, native.sample_rate
            source = "synth"

        await self._store(
            key, req, provider, model, params_canon,
            audio, store_encoding, store_rate, existing=existing, replace=True,
        )
        if source == "synth":
            await self._metrics.record_metrics(creates=1, synth_calls=1)
        else:
            await self._metrics.record_metrics(creates=1, base64_uploads=1)

        status = "OVERRIDDEN" if overridden else "CREATED"
        logger.info(
            f"CREATE {status} source={source} key={key[:12]}… "
            f"provider={provider} size={len(audio)}B"
        )
        return key, status, source, len(audio), provider, model, store_encoding, store_rate

    async def warm_split(self, req: TTSRequest) -> int:
        """Predictive warm: synth the phrase ONCE with word timestamps, then
        slice its audio into every contiguous sub-phrase and store each as a
        native cache entry (check-before-store, so no re-synth).

        One Cartesia call per phrase; the resulting substring-closed set of
        entries is what the binary-search segmentation on read relies on. If the
        provider can't supply timestamps or Cartesia's word boundaries don't
        align with the input words, falls back to storing just the full phrase.

        Returns the number of (sub-)phrase entries stored.
        """
        provider, model = parse_model_id(req.model_id)
        instance = self._get_provider(provider)
        if instance is None:
            raise ProviderNotConfigured(provider)

        norm_words = normalize_text(req.transcript).split()
        if not norm_words:
            return 0
        _, _, _, _, key = self._resolve(req)

        # Synth once; prefer timestamps so we can split. Fallback: full phrase.
        # (We synth even if the full phrase is already cached: normal caching
        # stores audio WITHOUT word boundaries, so we need a timestamped synth to
        # slice sub-phrases. The tracker's warmed-set ensures one warm per phrase.)
        aligned = False
        starts: list[float] = []
        gate = get_gate(provider)
        try:
            if hasattr(instance, "synth_with_timestamps"):
                async with gate:
                    audio, cwords, starts = await instance.synth_with_timestamps(
                        text=req.transcript, voice_id=req.voice.id, model=model,
                        language=req.language, params=req.params,
                    )
                aligned = bool(cwords) and len(cwords) == len(norm_words)
            else:
                raise NotImplementedError
        except Exception as e:
            logger.warning(f"timestamped synth failed for warm ({e}); full-phrase store")
            result = await self._synthesize(req, provider, model)
            audio = result.audio

        rate, bps = 16000, 2
        clip_end = len(audio) / (rate * bps)

        def byte_off(t: float) -> int:
            return int(round(t * rate * bps))

        # Sub-phrase word-index ranges to store: every contiguous slice if
        # aligned, else just the whole phrase.
        n = len(norm_words)
        ranges = (
            [(i, j) for i in range(n) for j in range(i + 1, n + 1)]
            if aligned
            else [(0, n)]
        )

        stored = 0
        for i, j in ranges:
            if aligned:
                end_t = starts[j] if j < len(starts) else clip_end
                slc = audio[byte_off(starts[i]) : byte_off(end_t)]
            else:
                slc = audio
            sub_text = " ".join(norm_words[i:j])
            sub_req = TTSRequest(
                model_id=req.model_id,
                transcript=sub_text,
                voice=req.voice,
                language=req.language,
                output_format=req.output_format,
                params=req.params,
            )
            sp, sm, _of, spc, skey = self._resolve(sub_req)
            sexisting = await self._metadata.get(skey)
            if sexisting and not self._expired(sexisting):
                continue  # already have this sub-phrase (e.g. from a prior split)
            await self._store(
                skey, sub_req, sp, sm, spc, slc, "pcm_s16le", rate, existing=sexisting
            )
            stored += 1

        await self._metrics.record_metrics(creates=stored, synth_calls=1)
        logger.info(
            f"WARM-SPLIT key={key[:12]}… provider={provider} words={n} "
            f"stored={stored} aligned={aligned}"
        )
        return stored

    async def delete(self, req: TTSRequest):
        """Delete by derived key. Returns (deleted_bool, key).

        The store adjusts provider_totals atomically inside the DELETE
        transaction (using the row's actual size at delete time), so a concurrent
        override of the same key can't make totals drift."""
        provider, model, of, params_canon, key = self._resolve(req)
        record = await self._metadata.get(key)
        if not record:
            return False, key
        # storage_path is key-derived (content-addressed), so it's correct even
        # if a concurrent override rewrote the row's bytes between get and delete.
        if not await self._metadata.delete(key):
            return False, key
        await self._blobs.delete(record.storage_path)
        await self._metrics.record_metrics(deletes=1)
        logger.info(f"DELETE key={key[:12]}… provider={provider}")
        return True, key

    async def clear(self, provider: str | None = None, voice_id: str | None = None) -> int:
        """Delete all entries (optionally filtered). Returns count removed.

        The store adjusts provider_totals atomically inside the DELETE
        transaction (SELECT+DELETE in one txn, so a concurrent insert can't
        escape the clear)."""
        deleted = await self._metadata.delete_filtered(provider=provider, voice_id=voice_id)
        for _prov, _size, path in deleted:
            await self._blobs.delete(path)
        if deleted:
            await self._metrics.record_metrics(deletes=len(deleted))
        logger.info(f"CLEAR removed {len(deleted)} entries (provider={provider}, voice_id={voice_id})")
        return len(deleted)

    @property
    def session_stats(self) -> dict:
        """Ephemeral hit/miss counters since process start."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "requests": total,
            "hit_rate": round(self._hits / total, 4) if total else None,
        }
