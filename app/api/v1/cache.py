"""Cache admin endpoints — list, fetch, and delete by raw key (for browsing)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.audio.format import content_type_for
from app.schemas.cache import CacheEntryInfo, PaginatedCache

router = APIRouter()


def _to_info(r) -> CacheEntryInfo:
    return CacheEntryInfo(
        key=r.key,
        provider=r.provider,
        voice_id=r.voice_id,
        model=r.model,
        language=r.language,
        container=r.container,
        encoding=r.encoding,
        sample_rate=r.sample_rate,
        size_bytes=r.size_bytes,
        hit_count=r.hit_count,
        created_at=r.created_at,
    )


MAX_CACHE_LIST_LIMIT = 1000  # hard cap so a list call can never stream the whole DB


@router.get("/cache", response_model=PaginatedCache)
async def list_cache(
    request: Request,
    provider: str | None = None,
    voice_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Paginated entry listing. limit is clamped to [1, MAX_CACHE_LIST_LIMIT]."""
    metadata = request.app.state.metadata
    limit = max(1, min(limit, MAX_CACHE_LIST_LIMIT))
    offset = max(0, offset)
    # Fetch one extra row to detect a next page without a COUNT(*) scan.
    records = await metadata.list(
        provider=provider, voice_id=voice_id, limit=limit + 1, offset=offset
    )
    has_next = len(records) > limit
    entries = [_to_info(r) for r in records[:limit]]
    return PaginatedCache(entries=entries, offset=offset, limit=limit, has_next=has_next)


@router.get("/stats")
async def stats(
    request: Request,
    from_date: str | None = Query(default=None, alias="from"),
    to_date: str | None = Query(default=None, alias="to"),
):
    """Cache metrics. Request metrics are date-filterable (?from=&to= YYYY-MM-DD);
    the cache snapshot is the current point-in-time state. All from stored
    rollups/totals — no full-table scan per call."""
    for label, value in (("from", from_date), ("to", to_date)):
        if value is not None:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"invalid {label}; use YYYY-MM-DD"
                )
    metadata = request.app.state.metadata
    registry = request.app.state.registry
    metrics = await metadata.metrics_summary(from_date=from_date, to_date=to_date)
    snapshot = await metadata.stats()
    return {
        "range": {"from": from_date, "to": to_date},
        **metrics,
        "entries": snapshot["entries"],
        "total_bytes": snapshot["total_bytes"],
        "by_provider": snapshot["by_provider"],
        "providers_configured": registry.configured(),
        "session": request.app.state.cache.session_stats,
    }


@router.post("/cache/clear")
async def clear_cache(
    request: Request,
    provider: str | None = None,
    voice_id: str | None = None,
):
    """Delete all entries (optionally filtered by provider/voice_id)."""
    cache = request.app.state.cache
    count = await cache.clear(provider=provider, voice_id=voice_id)
    return {"status": "cleared", "deleted": count}


@router.get("/cache/{key}")
async def get_cache(key: str, request: Request):
    metadata = request.app.state.metadata
    blobs = request.app.state.blobs
    record = await metadata.get(key)
    if not record:
        raise HTTPException(status_code=404, detail="cache key not found")
    audio = await blobs.get(record.storage_path)
    return Response(
        content=audio,
        media_type=content_type_for(record.encoding),
        headers={"X-Provider": record.provider, "X-Voice-Id": record.voice_id},
    )


@router.delete("/cache/{key}")
async def delete_cache(key: str, request: Request):
    metadata = request.app.state.metadata
    blobs = request.app.state.blobs
    record = await metadata.get(key)
    if not record:
        raise HTTPException(status_code=404, detail="cache key not found")
    await metadata.delete(key)
    await blobs.delete(record.storage_path)
    return {"status": "deleted", "key": key}
