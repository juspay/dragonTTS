"""Regression tests for the pre-release critical fixes.

1. ResilienceGate must not leak its bulkhead semaphore under concurrent
   overlapping acquires. The gate is a cached singleton; the old shared
   ``_acquired`` flag was clobbered by overlapping holders, under-releasing the
   semaphore until the provider locked up with ProviderBusy. (This IS a live
   bug: ``semaphore.acquire()`` is a real await point, so holders genuinely
   overlap and race on the flag.)
2. /tts/create and /tts/create/bulk must map ProviderError -> 502 (and the bulk
   path captures it per-item instead of 500-ing the whole batch).
"""

from __future__ import annotations

import asyncio

import pytest

from app.cache.resilience import ResilienceGate
from app.providers.base import BaseTTSProvider, ProviderError


async def test_resilience_gate_no_leak_under_concurrent_overlap():
    """Overlapping acquire/release cycles must fully replenish the semaphore.

    Before the fix the shared ``_acquired`` flag meant each overlapping pair
    under-released by one permit; after enough cycles the bulkhead drained to
    zero and every later ``async with gate`` raised ProviderBusy. With
    per-acquire balancing, capacity is exact and stable."""
    gate = ResilienceGate(max_concurrent=4, rate_per_sec=0.0, wait_timeout=1.0)
    sem = gate._sem
    assert sem is not None

    async def cycle() -> None:
        async with gate:
            await asyncio.sleep(0.005)  # force overlap across the 4 permits

    # Many rounds of 8 overlapping cycles (more than the 4 permits) — enough to
    # drain the semaphore to zero under the old shared-flag bug.
    for _ in range(30):
        await asyncio.gather(*(cycle() for _ in range(8)))

    # Semaphore fully replenished: exactly 4 permits available, no more, no less.
    for _ in range(4):
        await asyncio.wait_for(sem.acquire(), timeout=0.5)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sem.acquire(), timeout=0.05)


async def test_resilience_gate_release_on_partial_acquire():
    """If the rate-limiter raises after the bulkhead was acquired, the bulkhead
    permit is released (no leak from a partial acquire).

    We inject a bucket that always raises: acquire() takes the single bulkhead
    permit, then the bucket raises, so acquire() must release the bulkhead
    before propagating — else the provider locks up."""
    gate = ResilienceGate(max_concurrent=1, rate_per_sec=0.0, wait_timeout=1.0)

    class _BoomBucket:
        async def acquire(self, n: float = 1.0) -> None:
            raise RuntimeError("bucket exploded")

    gate._bucket = _BoomBucket()

    with pytest.raises(RuntimeError):
        await gate.acquire()

    # Bulkhead permit was released back -> re-acquire succeeds promptly.
    await asyncio.wait_for(gate._sem.acquire(), timeout=0.5)


class _ErrorProvider(BaseTTSProvider):
    """Always raises ProviderError — exercises the 502 mapping."""

    name = "cartesia"

    async def synth(self, *, text, voice_id, model, language, params):
        raise ProviderError("simulated upstream failure")

    async def stream_synth(self, *, text, voice_id, model, language, params):
        raise ProviderError("simulated upstream failure")
        yield  # pragma: no cover  (makes it an async generator)


def test_create_maps_provider_error_to_502(app_client, pcm_request):
    app_client.app.state.registry._providers["cartesia"] = _ErrorProvider()
    resp = app_client.post("/tts/create", json=pcm_request)
    assert resp.status_code == 502
    assert "upstream" in resp.json()["detail"]


def test_create_bulk_captures_provider_error_per_item(app_client, pcm_request):
    app_client.app.state.registry._providers["cartesia"] = _ErrorProvider()
    resp = app_client.post("/tts/create/bulk", json=[pcm_request, pcm_request])
    assert resp.status_code == 200  # per-item errors, not a batch-wide 500
    body = resp.json()
    assert body["created"] == 0
    assert body["errors"] == 2
