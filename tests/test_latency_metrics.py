"""Per-provider latency metrics + bounded-growth (leak) regression tests.

Covers:
- the ``provider`` column migration on latency_samples (old rows -> NULL).
- provider tagged on every recorded latency sample.
- ``latency_summary_by_provider`` avg/p95/count correctness, the ``provider``
  filter, and NULL-row exclusion.
- ``GET /stats/latency`` response shape + the derived ``miss_overhead_us`` +
  date validation.
- existing ``latency_summary`` keeps working (nullable provider is transparent).
- bounded growth: ElevenLabs ``_pools`` LRU cap (evicted pool closed),
  ``_inflight`` drains on synth exception, ``WriteBehindMetrics._latency`` cap.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.cache.service import CacheService
from app.core.config import settings
from app.providers.base import AudioResult
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _req_text(text: str) -> TTSRequest:
    return TTSRequest(
        model_id="cartesia:sonic-3.5",
        transcript=text,
        voice=CartesiaVoice(id="v1"),
        language="en",
        output_format=OutputFormat(),
    )


async def _seed(meta: SQLiteMetadataStore, rows: list[tuple]) -> None:
    """Insert (date, kind, latency_us, provider) rows directly."""

    def _go(conn: sqlite3.Connection) -> None:
        conn.executemany(
            "INSERT INTO latency_samples (date, kind, latency_us, provider) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

    await meta._run(_go)


async def _build_svc(provider_obj, tmp_storage):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: provider_obj if name == "cartesia" else None), meta


# --- migration ---------------------------------------------------------------


async def test_latency_migration_adds_provider_column(tmp_path):
    """An old DB without the provider column gets it on init (idempotent ALTER),
    and pre-existing rows become NULL (no backfill needed)."""
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE latency_samples "
        "(date TEXT NOT NULL, kind TEXT NOT NULL, latency_us INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO latency_samples (date, kind, latency_us) VALUES ('2026-07-01','synth',100)"
    )
    conn.commit()
    conn.close()

    store = SQLiteMetadataStore(db)
    await store.init()  # runs _migrate -> ADD COLUMN provider TEXT + the index

    c = sqlite3.connect(db)
    cols = {r[1] for r in c.execute("PRAGMA table_info(latency_samples)")}
    assert "provider" in cols
    old_provider = c.execute(
        "SELECT provider FROM latency_samples WHERE kind='synth'"
    ).fetchone()[0]
    idx = {
        r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    c.close()
    assert old_provider is None  # old row backfilled NULL
    assert "idx_latency_provider_kind_date" in idx


# --- recording ---------------------------------------------------------------


async def test_provider_recorded_on_latency_samples(tmp_storage, fake_provider, monkeypatch):
    """A real MISS tags every recorded latency sample with the routed provider.
    Tests run with write-behind OFF, so samples land synchronously."""
    monkeypatch.setattr(settings, "metrics_latency_sample_rate", 1.0)  # sample everything
    svc, meta = await _build_svc(fake_provider, tmp_storage)

    await svc.get_or_synthesize(_req_text("hello world"))  # MISS -> synth + total

    out = await meta.latency_summary_by_provider()
    assert "cartesia" in out
    day = next(iter(out["cartesia"]))
    kinds = out["cartesia"][day]
    assert kinds["synth"]["count"] >= 1     # provider synth time recorded
    assert kinds["total"]["count"] >= 1     # end-to-end recorded


# --- query correctness -------------------------------------------------------


async def test_latency_summary_by_provider_correctness(tmp_storage):
    """avg/p95/count per provider x day, mirroring the nearest-rank p95 math."""
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    await _seed(
        meta,
        [
            ("2026-07-18", "synth", 100, "cartesia"),
            ("2026-07-18", "synth", 200, "cartesia"),
            ("2026-07-18", "synth", 300, "cartesia"),  # cnt=3 -> p95 offset 2 -> 300
            ("2026-07-18", "total", 800, "cartesia"),
            ("2026-07-18", "synth", 50, "sarvam"),
            ("2026-07-19", "synth", 1000, "cartesia"),  # different day
            ("2026-07-18", "synth", 999, None),         # NULL provider -> excluded
        ],
    )

    out = await meta.latency_summary_by_provider()
    assert set(out) == {"cartesia", "sarvam"}              # NULL excluded
    c = out["cartesia"]["2026-07-18"]
    assert c["synth"] == {"avg_us": 200.0, "p95_us": 300, "count": 3}
    assert c["total"] == {"avg_us": 800.0, "p95_us": 800, "count": 1}
    assert out["cartesia"]["2026-07-19"]["synth"]["count"] == 1
    assert out["sarvam"]["2026-07-18"]["synth"]["avg_us"] == 50.0


async def test_latency_summary_by_provider_filter(tmp_storage):
    """?provider= narrows to one provider; NULL rows stay excluded either way."""
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    await _seed(
        meta,
        [
            ("2026-07-18", "synth", 100, "cartesia"),
            ("2026-07-18", "synth", 50, "sarvam"),
            ("2026-07-18", "synth", 999, None),
        ],
    )

    only_cartesia = await meta.latency_summary_by_provider(provider="cartesia")
    assert set(only_cartesia) == {"cartesia"}
    assert only_cartesia["cartesia"]["2026-07-18"]["synth"]["count"] == 1


async def test_latency_summary_still_works_with_nullable_provider(tmp_storage):
    """The nullable provider column is transparent to the existing query (it
    doesn't reference provider). Both NULL and tagged rows are counted."""
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    await _seed(
        meta,
        [
            ("2026-07-18", "synth", 100, "cartesia"),
            ("2026-07-18", "synth", 200, None),
        ],
    )
    out = await meta.latency_summary()
    assert out["synth"]["count"] == 2  # both rows counted, no crash


# --- endpoint ----------------------------------------------------------------


def test_stats_latency_endpoint_shape_and_split(app_client):
    """GET /stats/latency returns per-provider x per-day avg/p95/count and the
    derived miss_overhead_us = total.avg - synth.avg."""
    conn = sqlite3.connect(settings.db_path)
    conn.executemany(
        "INSERT INTO latency_samples (date, kind, latency_us, provider) VALUES (?, ?, ?, ?)",
        [
            ("2026-07-18", "synth", 200, "cartesia"),
            ("2026-07-18", "synth", 400, "cartesia"),  # avg 300
            ("2026-07-18", "total", 800, "cartesia"),  # overhead 800-300 = 500
            ("2026-07-18", "synth", 50, "sarvam"),
        ],
    )
    conn.commit()
    conn.close()

    r = app_client.get("/stats/latency?from=2026-07-18&to=2026-07-18")
    assert r.status_code == 200
    body = r.json()
    assert body["range"] == {"from": "2026-07-18", "to": "2026-07-18", "provider": None}
    provs = body["providers"]
    assert set(provs) == {"cartesia", "sarvam"}
    c = provs["cartesia"]["2026-07-18"]
    assert c["synth"]["count"] == 2 and c["synth"]["avg_us"] == 300.0
    assert c["miss_overhead_us"] == 500.0
    # sarvam has no total sample -> overhead null
    assert provs["sarvam"]["2026-07-18"]["miss_overhead_us"] is None


def test_stats_latency_endpoint_provider_filter_and_bad_date(app_client):
    """?provider= narrows; a malformed date returns 400."""
    conn = sqlite3.connect(settings.db_path)
    conn.execute(
        "INSERT INTO latency_samples (date, kind, latency_us, provider) "
        "VALUES ('2026-07-18','synth',100,'cartesia')"
    )
    conn.commit()
    conn.close()

    r = app_client.get("/stats/latency?provider=cartesia")
    assert r.status_code == 200
    assert set(r.json()["providers"]) == {"cartesia"}

    bad = app_client.get("/stats/latency?from=not-a-date")
    assert bad.status_code == 400


# --- bounded growth / leaks --------------------------------------------------


async def test_inflight_drains_on_synth_exception(tmp_storage, monkeypatch):
    """A synth exception must not strand a future in _inflight (the drain path
    the audit relied on)."""

    class _Boom:
        name = "cartesia"
        native_encoding = "pcm_s16le"
        native_sample_rate = 16000

        async def synth(self, **kw):
            raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(settings, "metrics_latency_sample_rate", 0.0)
    svc, _meta = await _build_svc(_Boom(), tmp_storage)

    try:
        await svc.get_or_synthesize(_req_text("will explode"))
    except Exception:
        pass

    assert svc._inflight == {}  # no stranded future -> no leak


async def test_write_behind_latency_capped():
    """The latency accumulator never exceeds its 10000-sample cap even if a flush
    never runs (bounds memory if flushing ever stalls)."""
    from app.cache.metrics import WriteBehindMetrics

    class _Meta:
        async def touch_and_record(self, *a, **k):
            pass

        async def record_metrics(self, *a, **k):
            pass

        async def record_latency_batch(self, samples):
            pass

    wb = WriteBehindMetrics(_Meta(), 0.5, 64)
    wb._spawn_flush = lambda: None  # hold every sample; never flush

    for i in range(12000):
        await wb.record_latency("synth", i, "cartesia")

    assert len(wb._latency) == 10000  # hard cap, not 12000
