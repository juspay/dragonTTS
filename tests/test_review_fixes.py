"""Regression tests for the post-review hardening.

Covers:
- ``replace_with_totals``: atomic REPLACE + provider_totals delta. The override/
  refresh path used to call ``put`` + ``adjust_totals`` separately, so a concurrent
  ``delete`` could drift totals. Now it re-reads the row under one txn.
- ``reconcile_blobs``: orphan blob files (no metadata row — left by a crash
  between a row-delete commit and its blob unlink) are reaped; live blobs kept.
- ``iter_blobs`` / ``checkpoint``: the helpers backing reconcile / WAL compaction.
- sarvam ``synth``: an empty upstream body now raises ``ProviderError`` (-> 502),
  not a bare ``Exception`` (-> 500).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from app.storage.base import CacheRecord
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _record(key: str = "k1", size: int = 100, *, path: str | None = None) -> CacheRecord:
    return CacheRecord(
        key=key, provider="cartesia", voice_id="v", model="m", language="en",
        params="", text="t", container="raw", encoding="pcm_s16le",
        sample_rate=16000, size_bytes=size, storage_path=path or f"a/b/{key}",
    )


async def _fresh_stores(tmp_storage):
    metadata = SQLiteMetadataStore(settings.db_path)
    await metadata.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return metadata, blobs


# --- replace_with_totals -------------------------------------------------


async def test_replace_with_totals_new_row_counts_entry(tmp_storage):
    metadata, _blobs = await _fresh_stores(tmp_storage)
    await metadata.replace_with_totals(_record(size=50))
    stats = await metadata.stats()
    assert stats["entries"] == 1
    assert stats["total_bytes"] == 50


async def test_replace_with_totals_existing_row_size_delta_only(tmp_storage):
    metadata, _blobs = await _fresh_stores(tmp_storage)
    await metadata.put_with_totals(_record(size=100))
    # Refresh/override the SAME key with a different size.
    await metadata.replace_with_totals(_record(size=250))
    stats = await metadata.stats()
    assert stats["entries"] == 1           # NOT +1 for replacing an existing row
    assert stats["total_bytes"] == 250     # 100 -> 250 (delta), not 100 + 250


async def test_replace_with_totals_two_distinct_keys(tmp_storage):
    metadata, _blobs = await _fresh_stores(tmp_storage)
    await metadata.replace_with_totals(_record("a", size=10))
    await metadata.replace_with_totals(_record("b", size=20))
    stats = await metadata.stats()
    assert stats["entries"] == 2
    assert stats["total_bytes"] == 30


# --- reconcile_blobs -----------------------------------------------------


async def test_reconcile_removes_orphan_blobs(tmp_storage):
    from app.cache.service import CacheService

    metadata, blobs = await _fresh_stores(tmp_storage)
    cache = CacheService(metadata, blobs, lambda name: None)
    await cache.start()
    try:
        live_key = "ab" * 20
        live_path = await blobs.put(live_key, b"live")           # row + blob
        await metadata.put_with_totals(_record(live_key, size=4, path=live_path))
        orphan_key = "cd" * 20
        orphan_path = await blobs.put(orphan_key, b"orphan")     # blob, NO row

        removed = await cache.reconcile_blobs()
        assert removed == 1
        assert not (Path(settings.blob_dir) / orphan_path).exists()
        assert (Path(settings.blob_dir) / live_path).exists()
    finally:
        await cache.stop()


async def test_reconcile_noop_when_clean(tmp_storage):
    from app.cache.service import CacheService

    metadata, blobs = await _fresh_stores(tmp_storage)
    cache = CacheService(metadata, blobs, lambda name: None)
    await cache.start()
    try:
        live_path = await blobs.put("ab" * 20, b"live")
        await metadata.put_with_totals(_record("ab" * 20, size=4, path=live_path))
        assert await cache.reconcile_blobs() == 0
        assert (Path(settings.blob_dir) / live_path).exists()
    finally:
        await cache.stop()


# --- helpers -------------------------------------------------------------


async def test_iter_blobs_yields_stored(tmp_storage):
    _metadata, blobs = await _fresh_stores(tmp_storage)
    k1, k2 = "ab" * 20, "cd" * 20
    await blobs.put(k1, b"x")
    await blobs.put(k2, b"y")
    found = {(k, p) async for k, p in blobs.iter_blobs()}
    keys = {k for k, _ in found}
    assert k1 in keys and k2 in keys
    for k, p in found:  # rel_path is the sharded ab/cd/<key> form
        assert p == f"{k[:2]}/{k[2:4]}/{k}"


async def test_checkpoint_runs_clean(tmp_storage):
    metadata, _blobs = await _fresh_stores(tmp_storage)
    await metadata.put_with_totals(_record(size=8))
    await metadata.checkpoint()  # must not raise
    assert await metadata.get("k1") is not None  # still readable after checkpoint


# --- provider error mapping ---------------------------------------------


async def test_sarvam_synth_raises_provider_error_on_empty_audio():
    """A 200-OK-but-empty Sarvam body must raise ProviderError (-> 502), not the
    bare Exception it used to (-> unhandled 500)."""
    from app.providers.base import ProviderError
    from app.providers.sarvam import SarvamProvider

    provider = SarvamProvider(api_key="k")

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"audios": []}  # HTTP 200 but no audio payload

    class _Client:
        async def post(self, *a, **k):
            return _Resp()

        async def aclose(self) -> None:
            pass

    provider._client = _Client()
    with pytest.raises(ProviderError):
        await provider.synth(
            text="hi", voice_id="v", model="bulbul:v2", language="en-IN", params={},
        )
