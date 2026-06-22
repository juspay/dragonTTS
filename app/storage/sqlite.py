"""SQLite metadata store (stdlib sqlite3, wrapped with asyncio.to_thread)."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import logger
from app.storage.base import CacheRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key              TEXT PRIMARY KEY,
    provider         TEXT NOT NULL,
    voice_id         TEXT NOT NULL,
    model            TEXT NOT NULL,
    language         TEXT NOT NULL,
    params           TEXT NOT NULL DEFAULT '',
    text             TEXT,
    container        TEXT NOT NULL,
    encoding         TEXT NOT NULL,
    sample_rate      INTEGER NOT NULL,
    size_bytes       INTEGER NOT NULL,
    storage_path     TEXT NOT NULL,
    hit_count        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    ttl_expires_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_provider_voice ON cache_entries(provider, voice_id);

-- Request metrics rolled up per UTC day (date-filterable via SUM, never a scan
-- of cache_entries).
CREATE TABLE IF NOT EXISTS metrics_daily (
    date            TEXT PRIMARY KEY,
    requests        INTEGER NOT NULL DEFAULT 0,
    hits            INTEGER NOT NULL DEFAULT 0,
    misses          INTEGER NOT NULL DEFAULT 0,
    bytes_served    INTEGER NOT NULL DEFAULT 0,
    synth_calls     INTEGER NOT NULL DEFAULT 0,
    base64_uploads  INTEGER NOT NULL DEFAULT 0,
    creates         INTEGER NOT NULL DEFAULT 0,
    deletes         INTEGER NOT NULL DEFAULT 0
);

-- Cache snapshot maintained incrementally so /stats is O(providers), not a
-- full-table GROUP BY.
CREATE TABLE IF NOT EXISTS provider_totals (
    provider    TEXT PRIMARY KEY,
    entries     INTEGER NOT NULL DEFAULT 0,
    total_bytes INTEGER NOT NULL DEFAULT 0
);
"""

_COLUMNS = (
    "key",
    "provider",
    "voice_id",
    "model",
    "language",
    "params",
    "text",
    "container",
    "encoding",
    "sample_rate",
    "size_bytes",
    "storage_path",
    "hit_count",
    "created_at",
    "last_accessed_at",
    "ttl_expires_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_record(row: sqlite3.Row) -> CacheRecord:
    d = dict(row)
    return CacheRecord(**{c: d[c] for c in _COLUMNS})


class SQLiteMetadataStore:
    """Async-friendly metadata store backed by a single SQLite file (WAL mode)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()

    def _new_connection(self) -> sqlite3.Connection:
        # One connection per OS thread. WAL permits many concurrent readers
        # across connections and a single serialized writer; busy_timeout makes
        # contending writers wait instead of raising "database is locked".
        # isolation_level=None => autocommit. Our ops are single-statement
        # (point lookups / single inserts); autocommit avoids implicit nested
        # transactions ("cannot start a transaction within a transaction") on
        # long-lived, reused thread connections.
        conn = sqlite3.connect(
            self.db_path, timeout=30, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA cache_size = -65536")  # ~64MB page cache
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap for reads
        return conn

    async def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        def _setup() -> None:
            conn = self._new_connection()
            try:
                conn.executescript(_SCHEMA)
                # Backfill snapshot totals if empty but entries exist (handles a
                # migration from before provider_totals existed).
                has_totals = conn.execute("SELECT COUNT(*) FROM provider_totals").fetchone()[0]
                if has_totals == 0:
                    conn.execute(
                        "INSERT INTO provider_totals (provider, entries, total_bytes) "
                        "SELECT provider, COUNT(*), COALESCE(SUM(size_bytes), 0) "
                        "FROM cache_entries GROUP BY provider "
                        "ON CONFLICT(provider) DO UPDATE SET "
                        "entries = excluded.entries, total_bytes = excluded.total_bytes"
                    )
            finally:
                conn.close()

        await asyncio.to_thread(_setup)
        logger.info(f"SQLite metadata store ready at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        """Lazily-created, thread-local connection (cached for reuse)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    async def get(self, key: str) -> CacheRecord | None:
        conn = self._conn()

        def _q() -> sqlite3.Row | None:
            return conn.execute(
                "SELECT * FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()

        row = await asyncio.to_thread(_q)
        return _row_to_record(row) if row else None

    async def put(self, record: CacheRecord) -> None:
        conn = self._conn()
        values = tuple(getattr(record, c) for c in _COLUMNS)
        placeholders = ",".join("?" for _ in _COLUMNS)

        def _w() -> None:
            conn.execute(
                f"INSERT OR REPLACE INTO cache_entries ({','.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                values,
            )
            conn.commit()

        await asyncio.to_thread(_w)

    async def touch(self, key: str) -> None:
        conn = self._conn()

        def _t() -> None:
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1, "
                "last_accessed_at = ? WHERE key = ?",
                (_now(), key),
            )
            conn.commit()

        await asyncio.to_thread(_t)

    async def delete(self, key: str) -> bool:
        conn = self._conn()

        def _d() -> int:
            cur = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount

        return (await asyncio.to_thread(_d)) > 0

    async def list(
        self,
        provider: str | None = None,
        voice_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CacheRecord]:
        conn = self._conn()
        clauses: list[str] = []
        args: list = []
        if provider:
            clauses.append("provider = ?")
            args.append(provider)
        if voice_id:
            clauses.append("voice_id = ?")
            args.append(voice_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        args.append(offset)

        def _q() -> list[sqlite3.Row]:
            return conn.execute(
                f"SELECT * FROM cache_entries{where} ORDER BY created_at DESC "
                f"LIMIT ? OFFSET ?",
                args,
            ).fetchall()

        rows = await asyncio.to_thread(_q)
        return [_row_to_record(r) for r in rows]

    async def delete_filtered(
        self, provider: str | None = None, voice_id: str | None = None
    ) -> list[tuple]:
        """Bulk-delete matching rows; return (provider, size_bytes, storage_path)
        per row so callers can adjust totals + delete blobs."""
        conn = self._conn()
        clauses: list[str] = []
        args: list = []
        if provider:
            clauses.append("provider = ?")
            args.append(provider)
        if voice_id:
            clauses.append("voice_id = ?")
            args.append(voice_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        def _d() -> list[tuple]:
            rows = conn.execute(
                f"SELECT provider, size_bytes, storage_path FROM cache_entries{where}", args
            ).fetchall()
            conn.execute(f"DELETE FROM cache_entries{where}", args)
            return [(r[0], r[1], r[2]) for r in rows]

        return await asyncio.to_thread(_d)

    async def adjust_totals(self, provider: str, delta_entries: int, delta_bytes: int) -> None:
        conn = self._conn()
        sql = (
            "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET "
            "entries = entries + excluded.entries, total_bytes = total_bytes + excluded.total_bytes"
        )
        await asyncio.to_thread(lambda: conn.execute(sql, (provider, delta_entries, delta_bytes)))

    async def record_metrics(self, **deltas: int) -> None:
        """Upsert today's UTC daily-rollup row, adding the given metric deltas."""
        if not deltas:
            return
        conn = self._conn()
        cols = list(deltas)
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        upd = ", ".join(f"{c} = {c} + excluded.{c}" for c in cols)
        sql = (
            f"INSERT INTO metrics_daily (date, {col_list}) VALUES (?, {placeholders}) "
            f"ON CONFLICT(date) DO UPDATE SET {upd}"
        )
        values = [datetime.now(timezone.utc).date().isoformat(), *(deltas[c] for c in cols)]
        await asyncio.to_thread(lambda: conn.execute(sql, values))

    async def touch_and_record(self, key: str, metric_deltas: dict) -> None:
        """Increment an entry's hit_count/last_accessed AND daily metrics in one hop."""
        conn = self._conn()
        cols = list(metric_deltas)
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        upd = ", ".join(f"{c} = {c} + excluded.{c}" for c in cols)
        msql = (
            f"INSERT INTO metrics_daily (date, {col_list}) VALUES (?, {placeholders}) "
            f"ON CONFLICT(date) DO UPDATE SET {upd}"
        )
        mvals = [datetime.now(timezone.utc).date().isoformat(), *(metric_deltas[c] for c in cols)]

        def _t() -> None:
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1, last_accessed_at = ? "
                "WHERE key = ?",
                (_now(), key),
            )
            conn.execute(msql, mvals)

        await asyncio.to_thread(_t)

    async def metrics_summary(self, from_date: str | None = None, to_date: str | None = None) -> dict:
        conn = self._conn()
        clauses: list[str] = []
        args: list = []
        if from_date:
            clauses.append("date >= ?")
            args.append(from_date)
        if to_date:
            clauses.append("date <= ?")
            args.append(to_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        def _q() -> dict:
            r = conn.execute(
                "SELECT COALESCE(SUM(requests),0), COALESCE(SUM(hits),0), "
                "COALESCE(SUM(misses),0), COALESCE(SUM(bytes_served),0), "
                "COALESCE(SUM(synth_calls),0), COALESCE(SUM(base64_uploads),0), "
                "COALESCE(SUM(creates),0), COALESCE(SUM(deletes),0) "
                f"FROM metrics_daily{where}",
                args,
            ).fetchone()
            requests, hits, misses, bytes_served, synth_calls, base64_uploads, creates, deletes = r
            return {
                "requests": requests,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / requests, 4) if requests else None,
                "bytes_served": bytes_served,
                "synth_calls": synth_calls,
                "base64_uploads": base64_uploads,
                "creates": creates,
                "deletes": deletes,
            }

        return await asyncio.to_thread(_q)

    async def stats(self) -> dict:
        """Cache snapshot from incrementally-maintained provider_totals (O(providers))."""
        conn = self._conn()

        def _q() -> dict:
            rows = conn.execute(
                "SELECT provider, entries, total_bytes FROM provider_totals"
            ).fetchall()
            by_provider = {r[0]: {"entries": r[1], "total_bytes": r[2]} for r in rows}
            return {
                "entries": sum(p["entries"] for p in by_provider.values()),
                "total_bytes": sum(p["total_bytes"] for p in by_provider.values()),
                "by_provider": by_provider,
            }

        return await asyncio.to_thread(_q)
