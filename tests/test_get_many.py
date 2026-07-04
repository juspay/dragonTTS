"""Batched key lookup (SQLiteMetadataStore.get_many).

Stitch probes every candidate sub-span in ONE query instead of N round-trips.
These lock in the contract: exact-key match, dedup, >500-key chunking under
SQLite's host-param limit, and that expired rows are returned (the caller
filters them -- get_many is a pure lookup).
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from app.core.config import settings
from app.storage.base import CacheRecord
from app.storage.sqlite import SQLiteMetadataStore


def _rec(key: str) -> CacheRecord:
    return CacheRecord(
        key=key, provider="cartesia", voice_id="v", model="m", language="en",
        params="", text=key, container="raw", encoding="pcm_s16le",
        sample_rate=16000, size_bytes=10, storage_path=f"ab/cd/{key}", hit_count=0,
        created_at="2026-01-01T00:00:00+00:00",
        last_accessed_at="2026-01-01T00:00:00+00:00", ttl_expires_at=None,
    )


@pytest.fixture
async def store(tmp_storage):
    s = SQLiteMetadataStore(settings.db_path)
    await s.init()
    return s


async def test_get_many_empty_and_falsy(store):
    assert await store.get_many([]) == {}
    # falsy keys (None / "") are dropped before the query -- no SQL errors.
    assert await store.get_many([None, "", None]) == {}


async def test_get_many_returns_only_present(store):
    await store.put(_rec("a"))
    await store.put(_rec("b"))
    out = await store.get_many(["a", "b", "missing"])
    assert set(out) == {"a", "b"}
    assert out["a"].key == "a" and out["b"].text == "b"


async def test_get_many_dedups_keys(store):
    await store.put(_rec("a"))
    out = await store.get_many(["a", "a", "a"])
    assert list(out) == ["a"]           # one row, not three


async def test_get_many_chunks_past_500(store):
    # > 500 keys must be split across IN-clauses (SQLite's param limit). All
    # present keys come back, with none lost at a chunk boundary.
    keys = [f"k{i:05d}" for i in range(1200)]
    await asyncio.gather(*(store.put(_rec(k)) for k in keys))
    out = await store.get_many(keys)
    assert len(out) == 1200
    assert all(k in out for k in keys)


async def test_get_many_does_not_filter_expired(store):
    # get_many is a pure lookup; expiry is the caller's concern (stitch filters).
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    await store.put(_rec("fresh"))
    expired = _rec("old")
    expired.ttl_expires_at = past
    await store.put(expired)
    out = await store.get_many(["fresh", "old"])
    assert set(out) == {"fresh", "old"}     # expired "old" still returned
