"""Tests for the /stats analytics expansion: words (cached/served/synthesized),
per-provider rollup, stitch counters, latency avg/p95, and schema migration."""

from __future__ import annotations

import sqlite3

import pytest

from app.cache.service import CacheService
from app.core.config import settings
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.base import CacheRecord
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _req(text: str) -> TTSRequest:
    return TTSRequest(
        model_id="cartesia:sonic-3.5", transcript=text, voice=CartesiaVoice(id="v1"),
        language="en", output_format=OutputFormat(), params={},
    )


@pytest.fixture
async def svc(tmp_storage, fake_provider):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None)


def _record(key="k1", text="one two three", size=100) -> CacheRecord:
    return CacheRecord(
        key=key, provider="cartesia", voice_id="v", model="m", language="en",
        params="", text=text, container="raw", encoding="pcm_s16le",
        sample_rate=16000, size_bytes=size, storage_path=f"a/b/{key}",
    )


# --- words cached (snapshot) -------------------------------------------


async def test_total_words_tracks_put_and_delete(tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    await meta.put_with_totals(_record("k1", "one two three"))
    s = await meta.stats()
    assert s["total_words"] == 3
    assert s["by_provider"]["cartesia"]["total_words"] == 3
    await meta.delete("k1")
    assert (await meta.stats())["total_words"] == 0


async def test_total_words_replace_size_and_word_delta(tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    await meta.put_with_totals(_record("k1", "one two", size=10))
    # Override same key with a different word count + size.
    await meta.replace_with_totals(_record("k1", "one two three four", size=20))
    s = await meta.stats()
    assert s["entries"] == 1            # not +1 on replace
    assert s["total_words"] == 4        # 2 -> 4 (delta), not 2+4
    assert s["total_bytes"] == 20


async def test_total_words_concurrent_same_key_no_drift(tmp_storage):
    import asyncio
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    rec = _record("k1", "one two three")
    # Concurrent identical inserts -> only one lands (INSERT OR IGNORE).
    await asyncio.gather(*(meta.put_with_totals(rec) for _ in range(6)))
    s = await meta.stats()
    assert s["entries"] == 1
    assert s["total_words"] == 3        # counted once, not 6x


# --- words served / synthesized (daily) --------------------------------


async def test_words_served_and_synthesized(svc):
    await svc.get_or_synthesize(_req("alpha beta gamma"))   # MISS
    await svc.get_or_synthesize(_req("alpha beta gamma"))   # HIT
    m = await svc._metadata.metrics_summary()
    assert m["requests"] == 2
    assert m["words_served"] == 6          # 3 words x 2 requests
    assert m["words_synthesized"] == 3     # only the MISS synthesized


# --- per-provider rollup ----------------------------------------------


async def test_per_provider_rollup(tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    await meta.record_metrics(provider="cartesia", requests=1, hits=1, words_served=5)
    await meta.record_metrics(
        provider="sarvam", requests=1, misses=1, synth_calls=1, words_served=3
    )
    pm = await meta.provider_metrics_summary()
    assert pm["cartesia"] == {
        "requests": 1, "hits": 1, "misses": 0, "synth_calls": 0,
        "bytes_served": 0, "words_served": 5, "hit_rate": 1.0,
    }
    assert pm["sarvam"]["misses"] == 1 and pm["sarvam"]["synth_calls"] == 1
    assert pm["sarvam"]["hit_rate"] == 0.0


# --- stitch analytics --------------------------------------------------


async def test_stitch_records_metrics(svc):
    # Seed a substring-closed prefix chain so stitch's binary search finds the
    # cached prefix (stitch assumes monotonicity under substring closure — if
    # "how are you" is cached, "how" and "how are" must be too).
    for sub in ("how", "how are", "how are you"):
        await svc.create(_req(sub))
    audio = await svc.stitch(_req("how are you today"), "cartesia", "sonic-3.5", "")
    assert audio is not None
    m = await svc._metadata.metrics_summary()
    assert m["stitch_calls"] == 1
    assert m["stitch_words_assembled"] == 3     # "how are you"
    assert m["stitch_words_synthesized"] == 1   # "today"
    assert m["stitch_coverage_avg"] == 0.75


# --- latency -----------------------------------------------------------


async def test_latency_recorded(svc, monkeypatch):
    monkeypatch.setattr(settings, "metrics_latency_sample_rate", 1.0)  # sample every request
    await svc.get_or_synthesize(_req("alpha beta gamma"))   # MISS
    await svc.get_or_synthesize(_req("alpha beta gamma"))   # HIT
    lat = await svc._metadata.latency_summary()
    assert lat["synth"]["count"] >= 1
    assert lat["total"]["count"] >= 2
    assert lat["cache_serve"]["count"] >= 1
    for kind in ("synth", "total", "cache_serve"):
        assert lat[kind]["p95_us"] >= lat[kind]["avg_us"]


async def test_stream_latency_recorded(svc, monkeypatch):
    """Streaming records end-to-end latency (total + ttfb) — including the live
    stream MISS path that previously had zero samples. stream() returns the
    generator before the request finishes, so the samples ride the generator
    lifecycle: ttfb on first byte, total in finally once it's consumed."""
    monkeypatch.setattr(settings, "metrics_latency_sample_rate", 1.0)

    # Live stream MISS (or non-native full-synth MISS — both are wrapped).
    # The samples only land once the generator is actually consumed.
    _, gen = await svc.stream(_req("alpha beta gamma"))
    assert b"".join([c async for c in gen])
    lat = await svc._metadata.latency_summary()
    assert lat["total"]["count"] >= 1
    assert lat["ttfb"]["count"] >= 1

    # Stream HIT (seeded): cache_serve is still recorded, plus total + ttfb.
    await svc.create(_req("beta gamma delta"))
    _, gen2 = await svc.stream(_req("beta gamma delta"))
    assert b"".join([c async for c in gen2])
    lat = await svc._metadata.latency_summary()
    assert lat["cache_serve"]["count"] >= 1
    for kind in ("total", "ttfb", "cache_serve"):
        assert lat[kind]["p95_us"] >= lat[kind]["avg_us"]


# --- migration on a pre-analytics DB ----------------------------------


async def test_migration_adds_columns_and_backfills_words(tmp_path):
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE cache_entries (
            key TEXT PRIMARY KEY, provider TEXT, voice_id TEXT, model TEXT,
            language TEXT, params TEXT DEFAULT '', text TEXT, container TEXT,
            encoding TEXT, sample_rate INTEGER, size_bytes INTEGER,
            storage_path TEXT, hit_count INTEGER DEFAULT 0, created_at TEXT,
            last_accessed_at TEXT, ttl_expires_at TEXT
        );
        CREATE TABLE metrics_daily (
            date TEXT PRIMARY KEY, requests INTEGER DEFAULT 0, hits INTEGER DEFAULT 0,
            misses INTEGER DEFAULT 0, bytes_served INTEGER DEFAULT 0,
            synth_calls INTEGER DEFAULT 0, base64_uploads INTEGER DEFAULT 0,
            creates INTEGER DEFAULT 0, deletes INTEGER DEFAULT 0
        );
        CREATE TABLE provider_totals (
            provider TEXT PRIMARY KEY, entries INTEGER DEFAULT 0, total_bytes INTEGER DEFAULT 0
        );
        CREATE INDEX idx_provider_voice ON cache_entries(provider, voice_id);
        """
    )
    conn.execute(
        "INSERT INTO cache_entries (key, provider, voice_id, model, language, params, text, "
        "container, encoding, sample_rate, size_bytes, storage_path, hit_count, created_at, "
        "last_accessed_at, ttl_expires_at) "
        "VALUES ('k','cartesia','v','m','en','','hello world','raw','pcm_s16le',16000,100,"
        "'a/b/k',0,'t','t',NULL)"
    )
    conn.execute(
        "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES ('cartesia', 1, 100)"
    )
    conn.commit()
    conn.close()

    store = SQLiteMetadataStore(db)
    await store.init()  # runs _migrate (adds columns) + total_words backfill

    c = sqlite3.connect(db)
    md_cols = {r[1] for r in c.execute("PRAGMA table_info(metrics_daily)")}
    pt_cols = {r[1] for r in c.execute("PRAGMA table_info(provider_totals)")}
    c.close()
    assert {
        "words_served", "words_synthesized", "stitch_calls",
        "stitch_words_assembled", "stitch_words_synthesized",
    } <= md_cols
    assert "total_words" in pt_cols
    # total_words backfilled from the seeded "hello world" (2 words)
    assert (await store.stats())["total_words"] == 2
