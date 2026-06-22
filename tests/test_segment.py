"""Binary-search segmentation — pure logic with a fake async is_cached."""

from __future__ import annotations

import pytest

from app.cache.segment import segment


def _make_is_cached(full_phrases: set[tuple[int, int]]):
    """Closure-aware is_cached(start, end): True if [start,end) is a contiguous
    sub-range of any cached full phrase — mirrors warm_split's substring closure."""
    async def is_cached(start: int, end: int) -> bool:
        return any(cl <= start and end <= ch for (cl, ch) in full_phrases)
    return is_cached


async def _seg(n, cached, **kw):
    return await segment(n, _make_is_cached(cached), **kw)


async def test_prefix_suffix_with_synth_middle():
    # words: [hi, nitya, sir]; cached: hi(0,1), sir(2,3). Middle "nitya" synth'd.
    segs = await _seg(3, {(0, 1), (2, 3)})
    assert segs == [(0, 1, True), (1, 2, False), (2, 3, True)]


async def test_nothing_cached_synths_whole():
    segs = await _seg(4, set())
    assert segs == [(0, 4, False)]


async def test_whole_cached():
    segs = await _seg(5, {(0, 5)})
    assert segs == [(0, 5, True)]


async def test_long_prefix():
    # words: [a, b, c, d, e]; cached prefixes a, ab, abc (closure). Prefix=3.
    cached = {(0, 1), (0, 2), (0, 3)}
    segs = await _seg(5, cached)
    # prefix 3, no suffix -> middle [3,5] (2 words <= SMALL) synth'd whole
    assert (0, 3, True) in segs
    # remainder is one synth span covering [3,5]
    synth = [s for s in segs if not s[2]]
    assert synth == [(3, 5, False)]


async def test_prefix_and_suffix_meet():
    # cached: prefix(0,3) and suffix(2,5) overlap -> whole covered
    cached = {(0, 1), (0, 2), (0, 3), (2, 5), (3, 5), (4, 5)}
    segs = await _seg(5, cached)
    # all spans cached
    assert all(c for _, _, c in segs)
    # spans tile [0,5]
    pos = 0
    for a, b, _ in segs:
        assert a == pos
        pos = b
    assert pos == 5


async def test_binary_search_uses_monotonicity():
    # Closure: cached(0,k) for k<=3, not for k>3. Binary search must find 3.
    cached = {(0, 1), (0, 2), (0, 3)}
    segs = await _seg(6, cached)
    cached_prefix = max((b for a, b, c in segs if c and a == 0), default=0)
    assert cached_prefix == 3


async def test_small_middle_not_split_further():
    # Big prefix, tiny 2-word middle with one cached word -> middle synth'd whole
    # (small threshold prevents splitting the 2-word middle into word fragments).
    cached = {(0, 1), (0, 2), (0, 3), (5, 6)}  # prefix 3, suffix 1, mid=[3,5]
    segs = await _seg(6, cached)
    spans = [(a, b, c) for a, b, c in segs]
    # prefix(0,3,T), middle(3,5,F) [not split despite word 5 boundary], suffix(5,6,T)
    assert (0, 3, True) in spans
    assert (5, 6, True) in spans
    assert (3, 5, False) in spans
    # No fragment inside the middle:
    assert (3, 4, False) not in spans
    assert (4, 5, False) not in spans
