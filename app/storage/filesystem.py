"""Filesystem blob store — content-addressed, sharded (ab/cd/<key>).

No in-process cache: hot reads are served by the OS page cache (kernel-managed,
self-invalidating), which makes an application-level LRU redundant — it only saved
~tens of μs over a page-cache hit, cost 64MB of heap, and its invalidation was a
correctness hazard. Bound the pod's memory with a k8s limit so the kernel caps the
page cache under pressure.

Writes ``fsync`` the file so the blob is at least as durable as the metadata row that
references it — ``cache/service._store`` writes the blob BEFORE committing metadata, so
a crash can never leave a row pointing at a missing/truncated blob.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

from app.core.logging import logger


class FilesystemBlobStore:
    def __init__(self, blob_dir: str):
        self.blob_dir = Path(blob_dir)
        # Shard dirs known to already exist; lets put() skip the mkdir syscall on
        # every write after the first per dir (mkdir is costly on a networked PVC).
        # Guarded by a lock because concurrent worker threads call put().
        self._seen_dirs: set[Path] = set()
        self._dirs_lock = threading.Lock()

    async def init(self) -> None:
        await asyncio.to_thread(self.blob_dir.mkdir, parents=True, exist_ok=True)
        logger.info(f"Filesystem blob store ready at {self.blob_dir}")

    def _rel_path(self, key: str) -> Path:
        return Path(key[:2]) / key[2:4] / key

    # -- store ops -----------------------------------------------------------

    async def put(self, key: str, data: bytes) -> str:
        rel = self._rel_path(key)

        def _write() -> str:
            path = self.blob_dir / rel
            parent = path.parent
            # mkdir is idempotent but a syscall; skip it once the shard dir is
            # known to exist. Double-checked so the lock is only taken on the
            # first create of each dir, not on every write.
            if parent not in self._seen_dirs:
                with self._dirs_lock:
                    if parent not in self._seen_dirs:
                        parent.mkdir(parents=True, exist_ok=True)
                        self._seen_dirs.add(parent)
            # fsync so the blob is at least as durable as the metadata row
            # committed after it. (Hot reads are served by the OS page cache, so
            # there's no app-level cache to keep coherent here.)
            with open(path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return str(rel)

        return await asyncio.to_thread(_write)

    async def get(self, storage_path: str) -> bytes:
        # No app-level cache: served straight from the OS page cache (hot) or disk.
        return await asyncio.to_thread((self.blob_dir / storage_path).read_bytes)

    async def delete(self, storage_path: str) -> bool:
        path = self.blob_dir / storage_path

        def _del() -> bool:
            if path.exists():
                path.unlink()
                return True
            return False

        return await asyncio.to_thread(_del)
