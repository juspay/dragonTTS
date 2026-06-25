"""Storage abstraction — metadata (DB) and audio blobs (filesystem) as protocols.

Today: SQLiteMetadataStore + FilesystemBlobStore.
Later: PostgresMetadataStore + GCSBlobStore — drop-in new implementations, no
change to the cache service or API layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CacheRecord:
    """One cached synthesis result. Metadata lives here; audio bytes in the blob store."""

    key: str
    provider: str
    voice_id: str
    model: str
    language: str
    params: str  # canonical JSON of tuning params ("" when none)
    text: str | None  # nullable, admin/debug only
    container: str
    encoding: str
    sample_rate: int
    size_bytes: int
    storage_path: str  # relative path within the blob store
    hit_count: int = 0
    created_at: str = ""
    last_accessed_at: str = ""
    ttl_expires_at: str | None = None


class MetadataStore(Protocol):
    async def get(self, key: str) -> CacheRecord | None: ...
    async def put(self, record: CacheRecord) -> None: ...
    async def touch(self, key: str) -> None: ...  # hit_count++, last_accessed_at=now
    async def touch_and_record(self, key: str, metric_deltas: dict) -> None: ...
    async def delete(self, key: str) -> bool: ...  # also adjusts provider_totals atomically
    async def delete_filtered(
        self, provider: str | None = None, voice_id: str | None = None
    ) -> list[tuple]: ...  # (provider, size_bytes, storage_path) per row; also adjusts totals atomically
    async def adjust_totals(self, provider: str, delta_entries: int, delta_bytes: int) -> None: ...
    async def record_metrics(self, **deltas: int) -> None: ...
    async def metrics_summary(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> dict: ...
    async def list(
        self,
        provider: str | None = None,
        voice_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CacheRecord]: ...
    async def stats(self) -> dict: ...


class BlobStore(Protocol):
    async def put(self, key: str, data: bytes) -> str: ...  # returns relative storage_path
    async def get(self, storage_path: str) -> bytes: ...
    async def delete(self, storage_path: str) -> bool: ...
