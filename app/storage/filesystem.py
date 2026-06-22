"""Filesystem blob store — content-addressed, sharded (ab/cd/<key>).

A byte-bounded in-memory LRU sits in front of the disk so hot cache hits skip
the file read entirely (no thread hop). The LRU is kept consistent because
``put`` pre-warms/overwrites and ``delete`` evicts — overrides reuse the same
content-addressed path, so stale entries can't survive.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger


class FilesystemBlobStore:
    def __init__(self, blob_dir: str, cache_bytes: int | None = None):
        self.blob_dir = Path(blob_dir)
        self._max_bytes = cache_bytes if cache_bytes is not None else settings.blob_cache_bytes
        self._lru: "OrderedDict[str, bytes]" = OrderedDict()
        self._size = 0

    async def init(self) -> None:
        await asyncio.to_thread(self.blob_dir.mkdir, parents=True, exist_ok=True)
        logger.info(
            f"Filesystem blob store ready at {self.blob_dir} "
            f"(in-memory LRU cap {self._max_bytes // (1024 * 1024)}MB)"
        )

    def _rel_path(self, key: str) -> Path:
        return Path(key[:2]) / key[2:4] / key

    # -- LRU (event-loop-thread only; single-threaded, no lock needed) --------

    def _cache_put(self, path: str, data: bytes) -> None:
        if path in self._lru:
            self._size -= len(self._lru[path])
        self._lru[path] = data
        self._size += len(data)
        self._lru.move_to_end(path)
        while self._size > self._max_bytes and self._lru:
            _, evicted = self._lru.popitem(last=False)
            self._size -= len(evicted)

    def _cache_get(self, path: str) -> bytes | None:
        data = self._lru.get(path)
        if data is not None:
            self._lru.move_to_end(path)
        return data

    def _cache_evict(self, path: str) -> None:
        if path in self._lru:
            self._size -= len(self._lru.pop(path))

    # -- store ops -----------------------------------------------------------

    async def put(self, key: str, data: bytes) -> str:
        rel = self._rel_path(key)

        def _write() -> str:
            path = self.blob_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return str(rel)

        storage_path = await asyncio.to_thread(_write)
        self._cache_put(storage_path, data)  # pre-warm: just written, likely read soon
        return storage_path

    async def get(self, storage_path: str) -> bytes:
        cached = self._cache_get(storage_path)
        if cached is not None:
            return cached
        data = await asyncio.to_thread((self.blob_dir / storage_path).read_bytes)
        self._cache_put(storage_path, data)
        return data

    async def delete(self, storage_path: str) -> bool:
        self._cache_evict(storage_path)
        path = self.blob_dir / storage_path

        def _del() -> bool:
            if path.exists():
                path.unlink()
                return True
            return False

        return await asyncio.to_thread(_del)
