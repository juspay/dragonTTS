"""SQLite metadata store (stdlib sqlite3, wrapped with asyncio.to_thread).

Each worker thread owns its own connection (thread-local), fetched INSIDE the
threaded call via :meth:`_run`. This is what makes the store safe and parallel
under concurrency: WAL then allows many concurrent readers across the
per-thread connections and a single serialized writer, and writes can't corrupt
the way they would if one event-loop-thread connection were shared across
workers (``check_same_thread=False`` + sqlite3 threadsafety=1 is unsafe for
concurrent use of a single connection).
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

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


T = TypeVar("T")


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
        # Match connect()'s timeout (30s): contending writers retry up to 30s
        # before raising SQLITE_BUSY.
        conn.execute("PRAGMA busy_timeout=30000")
        # Per-connection page cache. With one connection per worker thread,
        # keep this modest (8MB) so N connections stay memory-bounded; the
        # 256MB mmap below carries the real read working set (OS page cache).
        conn.execute("PRAGMA cache_size = -8192")  # ~8MB page cache
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
                # Compact any leftover -wal from a prior run now that the schema
                # is open (runtime growth is bounded by the periodic checkpoint).
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
            finally:
                conn.close()

        await asyncio.to_thread(_setup)
        logger.info(f"SQLite metadata store ready at {self.db_path}")

    async def checkpoint(self) -> None:
        """Run a TRUNCATE WAL checkpoint so the ``-wal`` file is compacted back
        into the main db. Passive auto-checkpoint (1000 frames) moves frames but
        doesn't shrink the ``-wal`` while worker connections stay open, so this
        is run on startup, periodically, and on shutdown to keep it bounded on
        the PVC. Best-effort: never fails a request over a checkpoint error."""

        def _cp(conn: sqlite3.Connection) -> None:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass

        await self._run(_cp)

    def _conn(self) -> sqlite3.Connection:
        """Lazily-created, thread-local connection (cached for reuse). MUST be
        called from the worker thread that will use it — see :meth:`_run`."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn(connection)`` on a worker thread using THAT thread's own
        connection. This is the concurrency-safe path: every worker thread gets
        its own SQLite connection, so reads parallelize under WAL and concurrent
        writes don't corrupt (no shared event-loop-thread connection)."""
        return await asyncio.to_thread(lambda: fn(self._conn()))

    async def get(self, key: str) -> CacheRecord | None:
        def _q(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return conn.execute(
                "SELECT * FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()

        row = await self._run(_q)
        return _row_to_record(row) if row else None

    async def put(self, record: CacheRecord) -> None:
        values = tuple(getattr(record, c) for c in _COLUMNS)
        placeholders = ",".join("?" for _ in _COLUMNS)
        sql = (
            f"INSERT OR REPLACE INTO cache_entries ({','.join(_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )

        def _w(conn: sqlite3.Connection) -> None:
            conn.execute(sql, values)
            conn.commit()

        await self._run(_w)

    async def put_with_totals(self, record: CacheRecord) -> None:
        """Insert a NEW cache row; bump provider_totals only if this call
        actually inserted it.

        ``INSERT OR IGNORE`` + rowcount is race-free under WAL: writers
        serialize, so concurrent stores of the SAME key produce exactly one
        insert and one totals bump — ``provider_totals`` can't drift. Each
        statement autocommits (no explicit BEGIN, which conflicts with
        isolation_level=None on reused thread-local connections). For overrides
        and expired-refresh use ``put`` (REPLACE) + ``adjust_totals``.
        """
        values = tuple(getattr(record, c) for c in _COLUMNS)
        placeholders = ",".join("?" for _ in _COLUMNS)
        insert_sql = (
            f"INSERT OR IGNORE INTO cache_entries ({','.join(_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        totals_sql = (
            "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES (?, 1, ?) "
            "ON CONFLICT(provider) DO UPDATE SET "
            "entries = entries + 1, total_bytes = total_bytes + excluded.total_bytes"
        )

        def _w(conn: sqlite3.Connection) -> None:
            # Atomic: the row insert and the provider_totals bump must both land
            # or neither. Under isolation_level=None each execute autocommits
            # independently, so wrap the pair in an explicit transaction.
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(insert_sql, values)
                if cur.rowcount == 1:  # newly inserted; a concurrent dup was ignored
                    conn.execute(totals_sql, (record.provider, record.size_bytes))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        await self._run(_w)

    async def replace_with_totals(self, record: CacheRecord) -> None:
        """REPLACE an existing row and adjust ``provider_totals`` atomically.

        Used by the refresh/override store path. ``put`` + ``adjust_totals``
        separately would not be atomic: a concurrent ``delete`` of the same key
        landing between them would drift ``provider_totals`` (delete decrements,
        then the stale adjust re-applies a delta against a row that's gone). The
        current row's size is re-read INSIDE the ``BEGIN IMMEDIATE`` transaction
        so the delta reflects the row's actual state under the write lock.
        ``provider`` can't change (it's part of the key), so only the size delta
        matters for an existing row; a missing prior row is a fresh insert (+1).
        """
        values = tuple(getattr(record, c) for c in _COLUMNS)
        placeholders = ",".join("?" for _ in _COLUMNS)
        replace_sql = (
            f"INSERT OR REPLACE INTO cache_entries ({','.join(_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        totals_sql = (
            "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET "
            "entries = entries + excluded.entries, total_bytes = total_bytes + excluded.total_bytes"
        )

        def _w(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN IMMEDIATE")
            try:
                prev = conn.execute(
                    "SELECT size_bytes FROM cache_entries WHERE key = ?", (record.key,)
                ).fetchone()
                conn.execute(replace_sql, values)
                if prev is None:
                    conn.execute(totals_sql, (record.provider, 1, record.size_bytes))
                else:
                    conn.execute(totals_sql, (record.provider, 0, record.size_bytes - prev[0]))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        await self._run(_w)

    async def touch(self, key: str) -> None:
        def _t(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1, "
                "last_accessed_at = ? WHERE key = ?",
                (_now(), key),
            )
            conn.commit()

        await self._run(_t)

    async def delete(self, key: str) -> bool:
        """Delete one row AND adjust provider_totals atomically.

        SELECT + DELETE + totals-adjust run in one ``BEGIN IMMEDIATE``
        transaction, so the captured size is the row's actual size at delete
        time (a concurrent override of the same key can't make totals drift)
        and a concurrent insert can't escape. Returns True iff a row was deleted.
        """
        totals_sql = (
            "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET "
            "entries = entries + excluded.entries, total_bytes = total_bytes + excluded.total_bytes"
        )

        def _d(conn: sqlite3.Connection) -> bool:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT provider, size_bytes FROM cache_entries WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return False
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.execute(totals_sql, (row[0], -1, -row[1]))
                conn.execute("COMMIT")
                return True
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        return await self._run(_d)

    async def list(
        self,
        provider: str | None = None,
        voice_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CacheRecord]:
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

        def _q(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                f"SELECT * FROM cache_entries{where} ORDER BY created_at DESC "
                f"LIMIT ? OFFSET ?",
                args,
            ).fetchall()

        rows = await self._run(_q)
        return [_row_to_record(r) for r in rows]

    async def all_keys(self) -> set[str]:
        """Every cache key — used to reap orphaned blob files."""

        def _q(conn: sqlite3.Connection) -> set[str]:
            return {row[0] for row in conn.execute("SELECT key FROM cache_entries")}

        return await self._run(_q)

    async def delete_filtered(
        self, provider: str | None = None, voice_id: str | None = None
    ) -> list[tuple]:
        """Bulk-delete matching rows AND adjust provider_totals atomically;
        return (provider, size_bytes, storage_path) per row for blob cleanup.

        SELECT + DELETE + per-provider totals-adjust run in one ``BEGIN
        IMMEDIATE`` transaction, so a concurrent insert can't escape the clear
        and totals can't drift.
        """
        clauses: list[str] = []
        args: list = []
        if provider:
            clauses.append("provider = ?")
            args.append(provider)
        if voice_id:
            clauses.append("voice_id = ?")
            args.append(voice_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        totals_sql = (
            "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET "
            "entries = entries + excluded.entries, total_bytes = total_bytes + excluded.total_bytes"
        )

        def _d(conn: sqlite3.Connection) -> list[tuple]:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    f"SELECT provider, size_bytes, storage_path FROM cache_entries{where}", args
                ).fetchall()
                if rows:
                    conn.execute(f"DELETE FROM cache_entries{where}", args)
                    # Sum (entries, bytes) deltas per provider, apply once each.
                    deltas: dict[str, list[int]] = {}
                    for r in rows:
                        d = deltas.setdefault(r[0], [0, 0])
                        d[0] -= 1
                        d[1] -= r[1]
                    for prov, (de, db) in deltas.items():
                        conn.execute(totals_sql, (prov, de, db))
                conn.execute("COMMIT")
                return [(r[0], r[1], r[2]) for r in rows]
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        return await self._run(_d)

    async def adjust_totals(self, provider: str, delta_entries: int, delta_bytes: int) -> None:
        sql = (
            "INSERT INTO provider_totals (provider, entries, total_bytes) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET "
            "entries = entries + excluded.entries, total_bytes = total_bytes + excluded.total_bytes"
        )

        def _w(conn: sqlite3.Connection) -> None:
            conn.execute(sql, (provider, delta_entries, delta_bytes))

        await self._run(_w)

    async def record_metrics(self, **deltas: int) -> None:
        """Upsert today's UTC daily-rollup row, adding the given metric deltas."""
        if not deltas:
            return
        cols = list(deltas)
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        upd = ", ".join(f"{c} = {c} + excluded.{c}" for c in cols)
        sql = (
            f"INSERT INTO metrics_daily (date, {col_list}) VALUES (?, {placeholders}) "
            f"ON CONFLICT(date) DO UPDATE SET {upd}"
        )
        values = [datetime.now(timezone.utc).date().isoformat(), *(deltas[c] for c in cols)]

        def _w(conn: sqlite3.Connection) -> None:
            conn.execute(sql, values)

        await self._run(_w)

    async def touch_and_record(self, key: str, metric_deltas: dict) -> None:
        """Increment an entry's hit_count/last_accessed AND daily metrics in one hop."""
        cols = list(metric_deltas)
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        upd = ", ".join(f"{c} = {c} + excluded.{c}" for c in cols)
        msql = (
            f"INSERT INTO metrics_daily (date, {col_list}) VALUES (?, {placeholders}) "
            f"ON CONFLICT(date) DO UPDATE SET {upd}"
        )
        mvals = [datetime.now(timezone.utc).date().isoformat(), *(metric_deltas[c] for c in cols)]

        def _t(conn: sqlite3.Connection) -> None:
            # Bump hit_count by the ``hits`` delta (default 1) so batched writes
            # (write-behind sums N hits into one call) advance the row by N, not 1.
            hits_delta = int(metric_deltas.get("hits", 1))
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + ?, last_accessed_at = ? "
                "WHERE key = ?",
                (hits_delta, _now(), key),
            )
            conn.execute(msql, mvals)

        await self._run(_t)

    async def metrics_summary(self, from_date: str | None = None, to_date: str | None = None) -> dict:
        clauses: list[str] = []
        args: list = []
        if from_date:
            clauses.append("date >= ?")
            args.append(from_date)
        if to_date:
            clauses.append("date <= ?")
            args.append(to_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        def _q(conn: sqlite3.Connection) -> dict:
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

        return await self._run(_q)

    async def stats(self) -> dict:
        """Cache snapshot from incrementally-maintained provider_totals (O(providers))."""

        def _q(conn: sqlite3.Connection) -> dict:
            rows = conn.execute(
                "SELECT provider, entries, total_bytes FROM provider_totals"
            ).fetchall()
            by_provider = {r[0]: {"entries": r[1], "total_bytes": r[2]} for r in rows}
            return {
                "entries": sum(p["entries"] for p in by_provider.values()),
                "total_bytes": sum(p["total_bytes"] for p in by_provider.values()),
                "by_provider": by_provider,
            }

        return await self._run(_q)
