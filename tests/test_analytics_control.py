"""Day-wise analytics + cache-control features: daily_stats, list text/date
filters, delete-by-text (exact/substring/dry-run), delete-by-age, clear preview."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.storage.base import CacheRecord


def _rec(
    key, text="t", provider="cartesia", voice="v1", age_days=0, size=100,
):
    created = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    return CacheRecord(
        key=key, provider=provider, voice_id=voice, model="m", language="en",
        params="", text=text, container="raw", encoding="pcm_s16le",
        sample_rate=16000, size_bytes=size, storage_path=key,
        created_at=created, last_accessed_at=created,
    )


async def _svc(tmp_storage, monkeypatch):
    from app.cache.service import CacheService
    from app.storage.filesystem import FilesystemBlobStore
    from app.storage.sqlite import SQLiteMetadataStore

    monkeypatch.setattr(settings, "enable_write_through", True)
    meta = SQLiteMetadataStore(settings.db_path)
    blobs = FilesystemBlobStore(settings.blob_dir)
    await meta.init()
    await blobs.init()
    return CacheService(meta, blobs, lambda name: None), meta


_MD_COLS = (
    "date,requests,hits,misses,bytes_served,synth_calls,base64_uploads,"
    "creates,deletes,words_served,words_synthesized,stitch_calls,"
    "stitch_words_assembled,stitch_words_synthesized"
)


# --- daily_stats -------------------------------------------------------------

async def test_daily_stats_shape_and_derived(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)

    def seed(conn):
        conn.execute(
            f"INSERT INTO metrics_daily({_MD_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-07-09", 10, 4, 6, 1000, 6, 0, 0, 0, 100, 40, 2, 30, 10),
        )
        conn.execute(
            "INSERT INTO metrics_daily_provider"
            "(date,provider,requests,hits,misses,synth_calls,bytes_served,words_served) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("2026-07-09", "cartesia", 7, 3, 4, 4, 600, 70),
        )
        conn.execute(
            "INSERT INTO metrics_daily_provider"
            "(date,provider,requests,hits,misses,synth_calls,bytes_served,words_served) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("2026-07-09", "elevenlabs", 3, 1, 2, 2, 400, 30),
        )

    await meta._run(seed)

    out = await svc.daily_stats(from_date="2026-07-09", to_date="2026-07-09")
    assert out["range"] == {"from": "2026-07-09", "to": "2026-07-09"}
    assert len(out["days"]) == 1
    day = out["days"][0]
    assert day["date"] == "2026-07-09"

    t = day["totals"]
    assert t["requests"] == 10 and t["hits"] == 4
    assert t["hit_rate"] == 0.4                       # 4/10
    assert t["words_from_cache_pct"] == 60            # (100-40)/100
    assert set(day["by_provider"]) == {"cartesia", "elevenlabs"}
    assert day["by_provider"]["cartesia"]["hit_rate"] == round(3 / 7, 4)
    # per-provider rows have no words_synthesized -> no words_from_cache_pct key
    assert "words_from_cache_pct" not in day["by_provider"]["cartesia"]


async def test_daily_stats_provider_filter_narrows_totals(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)

    def seed(conn):
        conn.execute(
            f"INSERT INTO metrics_daily({_MD_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-07-10", 100, 50, 50, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        conn.execute(
            "INSERT INTO metrics_daily_provider"
            "(date,provider,requests,hits,misses,synth_calls,bytes_served,words_served) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("2026-07-10", "gemini", 20, 8, 12, 12, 0, 0),
        )

    await meta._run(seed)

    out = await svc.daily_stats(provider="gemini")
    day = out["days"][0]
    # totals reflect gemini only (NOT the global 100/50)
    assert day["totals"]["requests"] == 20 and day["totals"]["hits"] == 8
    assert list(day["by_provider"]) == ["gemini"]


# --- list text/date filters --------------------------------------------------

async def test_list_q_exact_and_substring(tmp_storage, monkeypatch):
    _svc_obj, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("k1", text="hello world"))
    await meta.put_with_totals(_rec("k2", text="hello mate"))
    await meta.put_with_totals(_rec("k3", text="goodbye"))

    exact = await meta.list(q="hello world")
    assert {r.key for r in exact} == {"k1"}

    sub = await meta.list(q="hello", match="substring")
    assert {r.key for r in sub} == {"k1", "k2"}


async def test_list_created_date_filters(tmp_storage, monkeypatch):
    _svc_obj, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("old", age_days=10))
    await meta.put_with_totals(_rec("mid", age_days=3))
    await meta.put_with_totals(_rec("new", age_days=0))

    # created_after = 4 days ago -> mid + new
    cutoff = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
    after = await meta.list(created_after=cutoff, limit=100)
    assert {r.key for r in after} == {"mid", "new"}

    # created_before = 4 days ago -> old only (exclusive of that day)
    before = await meta.list(created_before=cutoff, limit=100)
    assert {r.key for r in before} == {"old"}


# --- delete_by_text ----------------------------------------------------------

async def test_delete_by_text_dry_run_then_real(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("k1", text="hello world"))
    await meta.put_with_totals(_rec("k2", text="hello mate"))

    # dry_run (default) -> preview, nothing deleted
    preview = await svc.delete_by_text(text="hello", match="substring")
    assert preview["matched"] == 2 and preview["deleted"] == 0 and preview["dry_run"] is True
    assert await meta.get("k1") is not None and await meta.get("k2") is not None

    # real delete
    real = await svc.delete_by_text(text="hello", match="substring", dry_run=False)
    assert real["matched"] == 2 and real["deleted"] == 2
    assert await meta.get("k1") is None and await meta.get("k2") is None


async def test_delete_by_text_exact_and_provider_filter(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("k1", text="dup", provider="cartesia"))
    await meta.put_with_totals(_rec("k2", text="dup", provider="gemini"))

    # exact text matches both providers; narrow to gemini
    out = await svc.delete_by_text(text="dup", provider="gemini", dry_run=False)
    assert out["deleted"] == 1
    assert await meta.get("k1") is not None   # cartesia kept
    assert await meta.get("k2") is None


# --- delete_by_age -----------------------------------------------------------

async def test_delete_by_age(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("old", age_days=30))
    await meta.put_with_totals(_rec("new", age_days=1))

    preview = await svc.delete_by_age(older_than_days=7)
    assert preview["matched"] == 1 and preview["deleted"] == 0
    assert await meta.get("old") is not None

    real = await svc.delete_by_age(older_than_days=7, dry_run=False)
    assert real["deleted"] == 1
    assert await meta.get("old") is None
    assert await meta.get("new") is not None


# --- clear_preview -----------------------------------------------------------

async def test_clear_preview_does_not_delete(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("k1", text="a", provider="cartesia", size=500))
    await meta.put_with_totals(_rec("k2", text="b", provider="gemini", size=300))

    preview = await svc.clear_preview()
    assert preview["would_delete"] == 2
    assert preview["bytes"] == 800
    assert preview["by_provider"]["cartesia"]["entries"] == 1
    assert preview["by_provider"]["gemini"]["bytes"] == 300
    # nothing actually deleted
    assert await meta.get("k1") is not None and await meta.get("k2") is not None

    filt = await svc.clear_preview(provider="gemini")
    assert filt["would_delete"] == 1 and filt["by_provider"]["gemini"]["entries"] == 1


# --- regressions surfaced by code review ------------------------------------

async def test_delete_by_text_keeps_provider_totals_correct(tmp_storage, monkeypatch):
    """Regression: totals_sql must DECREMENT (ADD), not overwrite (SET)."""
    svc, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("k1", text="hello world", size=100))
    await meta.put_with_totals(_rec("k2", text="hello mate", size=100))
    await meta.put_with_totals(_rec("k3", text="goodbye", size=100))
    assert (await meta.stats())["by_provider"]["cartesia"]["total_bytes"] == 300

    out = await svc.delete_by_text(text="hello", match="substring", dry_run=False)
    assert out["deleted"] == 2
    bp = (await meta.stats())["by_provider"]["cartesia"]
    assert bp["entries"] == 1
    assert bp["total_bytes"] == 100  # was overwritten with the negative delta (-200)


async def test_delete_by_age_keeps_provider_totals_correct(tmp_storage, monkeypatch):
    svc, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("old1", age_days=30, size=100))
    await meta.put_with_totals(_rec("old2", age_days=30, size=100))
    await meta.put_with_totals(_rec("new", age_days=1, size=100))

    await svc.delete_by_age(older_than_days=7, dry_run=False)
    bp = (await meta.stats())["by_provider"]["cartesia"]
    assert bp["entries"] == 1
    assert bp["total_bytes"] == 100


async def test_substring_escapes_like_wildcards(tmp_storage, monkeypatch):
    """Regression: '_' / '%' in user text must match literally, not as wildcards."""
    _svc_obj, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("und", text="a_b"))
    await meta.put_with_totals(_rec("acb", text="acb"))
    await meta.put_with_totals(_rec("axb", text="axb"))
    await meta.put_with_totals(_rec("pct", text="50%"))
    await meta.put_with_totals(_rec("p50", text="50X"))

    assert {r.key for r in await meta.list(q="a_b", match="substring")} == {"und"}
    assert {r.key for r in await meta.list(q="50%", match="substring")} == {"pct"}


async def test_delete_by_text_rejects_empty_filter(tmp_storage, monkeypatch):
    """Regression: no filter would DELETE the whole cache; must raise."""
    svc, _meta = await _svc(tmp_storage, monkeypatch)
    with pytest.raises(ValueError):
        await svc.delete_by_text()  # all of text/provider/voice_id None
    with pytest.raises(ValueError):
        await svc.delete_by_text(text=None, provider=None, voice_id=None, dry_run=False)


async def test_match_is_case_insensitive(tmp_storage, monkeypatch):
    _svc_obj, meta = await _svc(tmp_storage, monkeypatch)
    await meta.put_with_totals(_rec("k1", text="hello world"))
    await meta.put_with_totals(_rec("k2", text="goodbye"))
    # "SUBSTRING" (upper) should behave as substring, not silently fall to exact
    assert {r.key for r in await meta.list(q="hello", match="SUBSTRING")} == {"k1"}
    assert {r.key for r in await meta.list(q="hello", match="Substring")} == {"k1"}
    assert await meta.list(q="hello", match="exact") == []  # no exact "hello" text


def test_delete_by_text_endpoint_rejects_empty_body(app_client):
    """POST /cache/delete-by-text with no filters -> 400, not a full-cache wipe."""
    r = app_client.post("/cache/delete-by-text", json={"dry_run": False})
    assert r.status_code == 400
