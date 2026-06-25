"""Write-behind metrics: HIT touches + record_metrics are deferred off the read
path, then flushed on an interval / batch threshold / shutdown.

The autouse ``_sync_metrics_in_tests`` fixture in conftest forces
``settings.metrics_write_behind_enabled = False`` for every test, so each test
here flips it back on LOCALLY (and builds the CacheService AFTER, so
``self._metrics`` is the ``WriteBehindMetrics`` wrapper).
"""

from __future__ import annotations

import asyncio

import pytest

from app.cache.metrics import WriteBehindMetrics
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


def _req_text(text: str) -> TTSRequest:
    r = _req()
    r.transcript = text
    return r


async def _build_svc(tmp_storage, fake_provider, *, write_behind: bool):
    """Build a CacheService with write-behind ON or OFF (matches the svc fixture
    pattern in test_cache_service.py, but lets each test pick the metrics mode)."""
    settings.metrics_write_behind_enabled = write_behind
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None)
    await svc.start()
    return svc, meta


# -- deferral: counters NOT written after a HIT until a flush -----------------


async def test_hit_counters_deferred_until_stop(tmp_storage, fake_provider, monkeypatch):
    """With write-behind ON, N HITs leave the daily metrics row untouched
    (deferred), but ``await svc.stop()`` flushes them so the row holds the sum."""
    monkeypatch.setattr(settings, "metrics_write_behind_enabled", True)
    # Make the interval long so only stop() can flush within the test.
    monkeypatch.setattr(settings, "metrics_flush_interval_ms", 60_000)
    monkeypatch.setattr(settings, "metrics_flush_batch_size", 1_000)
    svc, meta = await _build_svc(tmp_storage, fake_provider, write_behind=True)
    assert isinstance(svc._metrics, WriteBehindMetrics)  # wrapper active

    # Seed the cache so subsequent reads are HITs that go through touch_and_record.
    await svc.create(_req())  # synth (calls=1); its record_metrics is also deferred
    # create()'s own record_metrics(creates=1, synth_calls=1) is pending too.

    # 3 HITs -> 3 deferred touch_and_record({requests,hits,bytes_served}).
    for _ in range(3):
        audio, h = await svc.get_or_synthesize(_req())
        assert h["X-Cache"] == "HIT"

    # Nothing flushed yet: the daily row is absent (or all-zero) because the
    # accumulators are still in-memory.
    before = await meta.metrics_summary()
    assert before["requests"] == 0
    assert before["hits"] == 0
    assert before["creates"] == 0

    # Graceful stop does a final drain -> all pending writes land.
    await svc.stop()

    after = await meta.metrics_summary()
    assert after["requests"] == 3          # 3 HIT touches
    assert after["hits"] == 3
    assert after["creates"] == 1           # the create() also flushed
    assert after["synth_calls"] == 1       # the create()'s synth
    # The cache row's hit_count also advanced (touches are part of the flush).
    _cached, record, *_ = await svc.check(_req())
    assert record is not None and record.hit_count == 3


async def test_hit_counters_deferred_until_interval(tmp_storage, fake_provider, monkeypatch):
    """The interval flusher drains pending writes without an explicit stop()."""
    monkeypatch.setattr(settings, "metrics_write_behind_enabled", True)
    monkeypatch.setattr(settings, "metrics_flush_interval_ms", 30)   # 30ms
    monkeypatch.setattr(settings, "metrics_flush_batch_size", 1_000)
    svc, meta = await _build_svc(tmp_storage, fake_provider, write_behind=True)

    await svc.create(_req())
    await svc.get_or_synthesize(_req())  # 1 HIT, deferred

    assert (await meta.metrics_summary())["hits"] == 0  # not yet flushed

    # Wait for at least one interval tick to fire the flusher.
    for _ in range(50):
        await asyncio.sleep(0.02)
        if (await meta.metrics_summary())["hits"] == 1:
            break

    assert (await meta.metrics_summary())["hits"] == 1  # interval flushed it
    await svc.stop()


# -- batch threshold: pending >= flush_batch triggers an immediate flush -------


async def test_batch_threshold_flushes_immediately(tmp_storage, fake_provider, monkeypatch):
    """Accumulating up to ``flush_batch`` pending writes flushes without waiting
    on the interval or stop()."""
    monkeypatch.setattr(settings, "metrics_write_behind_enabled", True)
    monkeypatch.setattr(settings, "metrics_flush_interval_ms", 60_000)
    monkeypatch.setattr(settings, "metrics_flush_batch_size", 2)  # tiny batch
    svc, meta = await _build_svc(tmp_storage, fake_provider, write_behind=True)

    await svc.create(_req())  # pending=1 (create's record_metrics)

    # The async batch-threshold flush is dispatched but may not have run yet;
    # yield to the loop so the create_task(self._flush()) completes.
    await svc.get_or_synthesize(_req())  # pending=2 -> triggers threshold flush
    # Let the dispatched flush task settle.
    for _ in range(50):
        await asyncio.sleep(0.005)
        if (await meta.metrics_summary())["requests"] >= 1:
            break

    snap = await meta.metrics_summary()
    assert snap["requests"] >= 1   # threshold flushed despite the 60s interval
    assert snap["hits"] >= 1
    await svc.stop()


# -- per-item failure is swallowed: one bad touch doesn't kill the flush loop --


async def test_per_item_failure_swallowed(tmp_storage, fake_provider, monkeypatch):
    """A failing touch_and_record for one key is logged and skipped; the summed
    record_metrics still lands, and the loop survives for the next flush."""
    monkeypatch.setattr(settings, "metrics_write_behind_enabled", True)
    monkeypatch.setattr(settings, "metrics_flush_interval_ms", 60_000)
    monkeypatch.setattr(settings, "metrics_flush_batch_size", 1_000)
    svc, meta = await _build_svc(tmp_storage, fake_provider, write_behind=True)

    real_touch = meta.touch_and_record

    call_state = {"n": 0}

    async def flaky_touch(key, deltas):
        call_state["n"] += 1
        if call_state["n"] == 1:
            raise RuntimeError("simulated DB blip on first touch")
        await real_touch(key, deltas)

    meta.touch_and_record = flaky_touch  # type: ignore[assignment]

    # First touch will raise inside _flush; it must be swallowed.
    await svc._metrics.touch_and_record("k-bad", {"requests": 1, "hits": 1, "bytes_served": 10})
    await svc._metrics.record_metrics(synth_calls=1)  # should still flush
    await asyncio.sleep(0.01)  # let any dispatched flush settle
    await svc._metrics._flush()  # force a drain

    # The summed record_metrics landed despite the per-item touch failure.
    assert (await meta.metrics_summary())["synth_calls"] == 1
    await svc.stop()


# -- synchronous fallback still works when write-behind is OFF -----------------


async def test_write_behind_disabled_is_synchronous(tmp_storage, fake_provider, monkeypatch):
    """Sanity: with write-behind OFF, ``self._metrics`` IS the raw metadata store
    and writes apply immediately (no flush needed). This is the path every other
    test relies on via the autouse fixture."""
    monkeypatch.setattr(settings, "metrics_write_behind_enabled", False)
    svc, meta = await _build_svc(tmp_storage, fake_provider, write_behind=False)
    assert svc._metrics is meta  # raw store, no wrapper

    await svc.create(_req())
    await svc.get_or_synthesize(_req())  # HIT -> synchronous touch

    snap = await meta.metrics_summary()
    assert snap["requests"] == 1   # applied immediately, no flush
    assert snap["hits"] == 1
    await svc.stop()
