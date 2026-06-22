"""Concurrency stress tests for SQLiteMetadataStore.

Validates that parallel hash lookups (and interleaved reads+writes) work
correctly against the shared single connection across asyncio worker threads —
i.e. SQLite + WAL handles concurrent access safely.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from app.storage.base import CacheRecord
from app.storage.sqlite import SQLiteMetadataStore


def _record(i: int) -> CacheRecord:
    now = datetime.now(timezone.utc).isoformat()
    key = f"key{i:06d}"
    return CacheRecord(
        key=key,
        provider="cartesia",
        voice_id="v",
        model="m",
        language="en",
        params="",
        text=None,
        container="raw",
        encoding="pcm_s16le",
        sample_rate=16000,
        size_bytes=100,
        storage_path=f"ab/cd/{key}",
        hit_count=0,
        created_at=now,
        last_accessed_at=now,
    )


async def test_concurrent_reads(tmp_path):
    store = SQLiteMetadataStore(str(tmp_path / "c.db"))
    await store.init()
    for i in range(200):
        await store.put(_record(i))

    keys = [f"key{i:06d}" for i in range(200)] * 3  # 600 concurrent reads
    start = time.perf_counter()
    results = await asyncio.gather(*(store.get(k) for k in keys))
    elapsed = time.perf_counter() - start

    assert all(r is not None for r in results)
    assert len(results) == len(keys)
    print(f"\n600 concurrent reads in {elapsed*1000:.1f} ms "
          f"= {len(keys)/elapsed:.0f} lookups/sec")


async def test_concurrent_reads_and_writes_interleaved(tmp_path):
    store = SQLiteMetadataStore(str(tmp_path / "m.db"))
    await store.init()

    async def writer(i):
        await store.put(_record(i))

    async def reader(i):
        return await store.get(f"key{i:06d}")

    # 150 writes + 150 reads fired concurrently on the shared connection
    tasks = [writer(i) for i in range(150)] + [reader(i) for i in range(150)]
    await asyncio.gather(*tasks)

    for i in range(150):
        assert await store.get(f"key{i:06d}") is not None
    print("\n150 concurrent writes + 150 concurrent reads: no errors, all readable")
