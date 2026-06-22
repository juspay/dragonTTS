"""Binary-search phrase segmentation for cache-backed stitching.

Given a word count ``n`` and an async ``is_cached(start, end)`` predicate — true
when ``words[start:end]`` is in the (substring-closed) cache — find the longest
cached prefix and suffix via binary search, then recurse on the middle. Small
middles are synthesized wholesale (avoid tiny low-prosody fragments).

The binary search relies on monotonicity: under substring closure,
``cached(words[0:k])`` implies ``cached(words[0:k-1])`` (a prefix of a cached
phrase is itself cached), so the cached-prefix lengths form a contiguous
``[1..b]`` range and a binary search finds ``b`` in O(log n) lookups. Same for
suffixes.

Returns a list of ``(start, end, is_cached)`` spans that tile ``[0, n)``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# Middles at or below this many words are synthesized wholesale (no further
# splitting) to avoid stitching tiny fragments with poor prosody.
SMALL = 3

IsCached = Callable[[int, int], Awaitable[bool]]


async def _longest_prefix(lo: int, hi: int, is_cached: IsCached) -> int:
    """Largest k in [0, hi-lo] with is_cached(lo, lo+k). Binary search; O(log n)."""
    best = 0
    left, right = 1, hi - lo
    while left <= right:
        mid = (left + right) // 2
        if await is_cached(lo, lo + mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


async def _longest_suffix(lo: int, hi: int, is_cached: IsCached) -> int:
    """Largest k in [0, hi-lo] with is_cached(hi-k, hi). Binary search; O(log n)."""
    best = 0
    left, right = 1, hi - lo
    while left <= right:
        mid = (left + right) // 2
        if await is_cached(hi - mid, hi):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


async def _segment_range(lo: int, hi: int, is_cached: IsCached, small: int) -> list[tuple[int, int, bool]]:
    n = hi - lo
    if n <= 0:
        return []

    p = await _longest_prefix(lo, hi, is_cached)
    s = await _longest_suffix(lo, hi, is_cached)

    if p == 0 and s == 0:
        # No cached edges -> synthesize this whole span.
        return [(lo, hi, await is_cached(lo, hi))]

    if p + s >= n:
        # Prefix and suffix cover/overlap the span; both cached.
        return [(lo, lo + p, True), (lo + p, hi, True)] if p < n else [(lo, hi, True)]

    spans: list[tuple[int, int, bool]] = []
    if p > 0:
        spans.append((lo, lo + p, True))
    mid_lo, mid_hi = lo + p, hi - s
    if mid_hi - mid_lo <= small:
        # Small middle: synthesize wholesale (or use if cached) rather than
        # splitting into tiny fragments with poor prosody.
        spans.append((mid_lo, mid_hi, await is_cached(mid_lo, mid_hi)))
    else:
        spans.extend(await _segment_range(mid_lo, mid_hi, is_cached, small))
    if s > 0:
        spans.append((hi - s, hi, True))
    return spans


async def segment(n: int, is_cached: IsCached, small: int = SMALL) -> list[tuple[int, int, bool]]:
    """Segment ``words[0:n]`` into cached/synth spans (binary search + recurse)."""
    return await _segment_range(0, n, is_cached, small)
