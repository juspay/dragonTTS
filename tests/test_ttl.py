"""Length-scaled TTL: per-phrase expiry scaling, the periodic purge, and the
one-shot backfill of pre-existing NULL-TTL entries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.storage.base import CacheRecord


def _set_ttl_knobs(monkeypatch, base=172800, per_word=21600, mx=864000):
    monkeypatch.setattr(settings, "cache_ttl_base_seconds", base)
    monkeypatch.setattr(settings, "cache_ttl_per_word_seconds", per_word)
    monkeypatch.setattr(settings, "cache_ttl_max_seconds", mx)


# --- the TTL function itself -------------------------------------------------

async def test_ttl_expires_at_scales_with_word_count_and_caps(monkeypatch):
    from app.cache.service import CacheService

    _set_ttl_knobs(monkeypatch)
    now = datetime.now(timezone.utc)

    def expiry(text):
        return datetime.fromisoformat(CacheService._ttl_expires_at(text))

    # 1 word -> base + 1*per_word
    e1 = expiry("hello")
    lo = now + timedelta(seconds=settings.cache_ttl_base_seconds + settings.cache_ttl_per_word_seconds - 5)
    hi = now + timedelta(seconds=settings.cache_ttl_base_seconds + settings.cache_ttl_per_word_seconds + 5)
    assert lo <= e1 <= hi

    # 10 words -> strictly later than 1 word
    e10 = expiry("one two three four five six seven eight nine ten")
    assert e10 > e1

    # absurd word count -> capped at max
    e_big = expiry("word " * 10000)
    cap_lo = now + timedelta(seconds=settings.cache_ttl_max_seconds - 5)
    cap_hi = now + timedelta(seconds=settings.cache_ttl_max_seconds + 5)
    assert cap_lo <= e_big <= cap_hi


async def test_ttl_expires_at_disabled_when_base_le_zero(monkeypatch):
    from app.cache.service import CacheService

    _set_ttl_knobs(monkeypatch, base=0)
    assert CacheService._ttl_expires_at("anything here") is None


# --- _store writes the length-scaled TTL -------------------------------------

async def test_store_sets_length_ttl(tmp_storage, fake_provider, monkeypatch):
    from app.cache.service import CacheService
    from app.schemas.tts import CartesiaVoice, TTSRequest
    from app.storage.filesystem import FilesystemBlobStore
    from app.storage.sqlite import SQLiteMetadataStore

    _set_ttl_knobs(monkeypatch)
    monkeypatch.setattr(settings, "enable_write_through", True)

    meta = SQLiteMetadataStore(settings.db_path)
    blobs = FilesystemBlobStore(settings.blob_dir)
    await meta.init()
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None)

    words = ["hello", "world", "foo"]  # 3 words
    req = TTSRequest(
        model_id="cartesia:sonic-3.5",
        transcript=" ".join(words),
        voice=CartesiaVoice(id="v1"),
        language="en",
    )
    await svc.get_or_synthesize(req)

    recs = await meta.list(limit=10)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.ttl_expires_at is not None

    exp = datetime.fromisoformat(rec.ttl_expires_at)
    now = datetime.now(timezone.utc)
    expected = now + timedelta(
        seconds=settings.cache_ttl_base_seconds + settings.cache_ttl_per_word_seconds * len(words)
    )
    assert abs((exp - expected).total_seconds()) < 5


# --- purge_expired -----------------------------------------------------------

def _rec(key, ttl, provider="cartesia", text="t"):
    return CacheRecord(
        key=key, provider=provider, voice_id="v1", model="m", language="en",
        params="", text=text, container="raw", encoding="pcm_s16le",
        sample_rate=16000, size_bytes=10, storage_path=key, ttl_expires_at=ttl,
    )


async def test_purge_expired_removes_only_expired(tmp_path):
    from app.storage.sqlite import SQLiteMetadataStore

    store = SQLiteMetadataStore(str(tmp_path / "purge.db"))
    await store.init()
    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    future = (now + timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    await store.put_with_totals(_rec("expired", past))
    await store.put_with_totals(_rec("fresh", future))
    await store.put_with_totals(_rec("permanent", None))

    rows = await store.purge_expired(now_iso)
    assert len(rows) == 1
    assert rows[0][2] == "expired"  # storage_path

    assert await store.get("expired") is None
    assert await store.get("fresh") is not None
    assert await store.get("permanent") is not None  # NULL ttl never purged


# --- backfill_missing_ttl ----------------------------------------------------

async def test_backfill_sets_random_range_and_is_idempotent(tmp_path):
    from app.storage.sqlite import SQLiteMetadataStore

    store = SQLiteMetadataStore(str(tmp_path / "backfill.db"))
    await store.init()
    for k in ("a", "b", "c"):
        await store.put_with_totals(_rec(k, None))

    n = await store.backfill_missing_ttl(48, 72)
    assert n == 3

    now = datetime.now(timezone.utc)
    for k in ("a", "b", "c"):
        r = await store.get(k)
        assert r.ttl_expires_at is not None
        exp = datetime.fromisoformat(r.ttl_expires_at)
        # randomized within [48h, 72h] from now (clock slack on both ends)
        assert now + timedelta(hours=47, minutes=-1) <= exp <= now + timedelta(hours=73)

    # Idempotent: no NULL rows remain -> 0 updated on a second run.
    n2 = await store.backfill_missing_ttl(48, 72)
    assert n2 == 0


async def test_backfill_leaves_already_set_rows_alone(tmp_path):
    from app.storage.sqlite import SQLiteMetadataStore

    store = SQLiteMetadataStore(str(tmp_path / "backfill2.db"))
    await store.init()
    fixed = (datetime.now(timezone.utc) + timedelta(hours=200)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    await store.put_with_totals(_rec("has_ttl", fixed))
    await store.put_with_totals(_rec("no_ttl", None))

    n = await store.backfill_missing_ttl(48, 72)
    assert n == 1  # only the NULL row

    assert (await store.get("has_ttl")).ttl_expires_at == fixed  # untouched


# --- API endpoint wiring -----------------------------------------------------

def test_backfill_ttl_endpoint(app_client):
    """Empty cache -> backfill is a no-op, endpoint still 200s."""
    r = app_client.post("/cache/backfill-ttl")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "backfilled"
    assert body["updated"] == 0


# --- regression: purge must DECREMENT provider_totals, not overwrite ---------

async def test_purge_expired_keeps_provider_totals_correct(tmp_path):
    from app.storage.sqlite import SQLiteMetadataStore

    store = SQLiteMetadataStore(str(tmp_path / "pt.db"))
    await store.init()
    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    future = (now + timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    await store.put_with_totals(_rec("e1", past))
    await store.put_with_totals(_rec("e2", past))
    await store.put_with_totals(_rec("e3", future))
    assert (await store.stats())["by_provider"]["cartesia"]["total_bytes"] == 30  # 3 x 10B

    await store.purge_expired(now.strftime("%Y-%m-%dT%H:%M:%S+00:00"))

    bp = (await store.stats())["by_provider"]["cartesia"]
    assert bp["entries"] == 1
    assert bp["total_bytes"] == 10  # regression: was overwritten with the negative delta (-20)


async def test_backfill_range_is_inclusive_of_max(tmp_path):
    """abs(random())%span with span=max-min+1 must be able to reach max_hours."""
    from app.storage.sqlite import SQLiteMetadataStore

    store = SQLiteMetadataStore(str(tmp_path / "bf.db"))
    await store.init()
    for k in range(2000):  # enough samples that max_hours is almost certainly hit
        await store.put_with_totals(_rec(f"k{k}", None))
    await store.backfill_missing_ttl(48, 72)
    now = datetime.now(timezone.utc)
    hours = []
    for k in range(2000):
        r = await store.get(f"k{k}")
        hours.append((datetime.fromisoformat(r.ttl_expires_at) - now).total_seconds() / 3600)
    assert min(hours) >= 48 - 1
    assert max(hours) <= 72 + 1
    assert max(hours) > 71  # the +1 span fix lets it reach ~72 (was capped at 71)
