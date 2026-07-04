"""DP segmentation — pure logic with an exact cached-span set (no closure)."""

from __future__ import annotations

from app.cache.segment import MAX_SPAN, segment_dp


def test_prefix_suffix_with_synth_middle():
    # words: [hi, nitya, sir]; cached: hi(0,1), sir(2,3). Middle "nitya" synth'd.
    assert segment_dp(3, {(0, 1), (2, 3)}) == [(0, 1, True), (1, 2, False), (2, 3, True)]


def test_nothing_cached_synths_whole():
    assert segment_dp(4, set()) == [(0, 4, False)]


def test_whole_cached():
    assert segment_dp(5, {(0, 5)}) == [(0, 5, True)]


def test_longest_prefix_without_closure_chain():
    # Only the 3-word prefix is cached -- NO need for the (0,1),(0,2) closure
    # chain the old binary search required. DP picks the longest cached span.
    assert segment_dp(5, {(0, 3)}) == [(0, 3, True), (3, 5, False)]


def test_middle_cached_span_found():
    # A cached span in the MIDDLE (not a prefix/suffix) is reused. The old
    # binary search needed substring closure to reach this; DP finds it directly
    # from an exact whole-phrase cache.
    # words: [hi, <name>, your, order]; cached: hi(0,1), your order(2,4)
    assert segment_dp(4, {(0, 1), (2, 4)}) == [(0, 1, True), (1, 2, False), (2, 4, True)]


def test_prefix_and_suffix_meet():
    # Adjacent cached spans tile [0,5) with no gap -> merged into one cached span.
    # (The old binary search "covered" [3,5) via an overlapping suffix under
    # closure; exact-match DP needs the span itself cached — adjacent, not
    # overlapping — which is the honest, no-slicing behavior.)
    cached = {(0, 3), (3, 5)}
    segs = segment_dp(5, cached)
    assert segs == [(0, 5, True)]


def test_consecutive_same_type_merged():
    assert segment_dp(4, {(0, 2), (2, 4)}) == [(0, 4, True)]   # adjacent cached -> one
    assert segment_dp(3, {(0, 1)}) == [(0, 1, True), (1, 3, False)]  # adjacent gaps -> one


def test_prefers_longer_cached_span_on_ties():
    segs = segment_dp(5, {(0, 2), (0, 3)})
    assert [(a, b) for a, b, c in segs if c] == [(0, 3)]   # picks the longer span


def test_personalized_template_split():
    # The motivating case: cached "Hi" + cached template; the name is the gap.
    # words: [Hi, Anarjit, your, order, is, six, hundred, ninety, nine]
    cached = {(0, 1), (2, 9)}              # "Hi" + "your order is ... ninety nine"
    segs = segment_dp(9, cached)
    assert segs == [(0, 1, True), (1, 2, False), (2, 9, True)]
    assert sum(hi - lo for lo, hi, c in segs if not c) == 1   # only the name synth'd


# --- edge cases ----------------------------------------------------------


def test_n_zero_and_one():
    assert segment_dp(0, set()) == []
    assert segment_dp(1, set()) == [(0, 1, False)]
    assert segment_dp(1, {(0, 1)}) == [(0, 1, True)]


def test_max_span_boundary():
    # a cached span of EXACTLY MAX_SPAN words is reused; one word longer is not
    # (the candidate j-range is capped at i+MAX_SPAN). Stitch checks the whole
    # phrase separately, so dropping a >MAX_SPAN span here is by design.
    assert segment_dp(MAX_SPAN, {(0, MAX_SPAN)}) == [(0, MAX_SPAN, True)]
    big = MAX_SPAN + 1
    segs = segment_dp(big, {(0, big)})
    assert all(not c for _, _, c in segs)                      # nothing reused
    assert sum(hi - lo for lo, hi, _ in segs) == big           # still a full tiling


def test_out_of_range_spans_are_ignored():
    # spans ending past n (or before 0) never match the candidate range and are
    # silently dropped, not crashed on.
    assert segment_dp(4, {(2, 5)}) == [(0, 4, False)]          # j=5 > n=4
    assert segment_dp(4, {(-1, 2)}) == [(0, 4, False)]         # i=-1 < 0


def test_all_cached_adjacent_merges_into_one():
    n = 6
    cached = {(i, i + 2) for i in range(0, n, 2)}  # (0,2),(2,4),(4,6)
    assert segment_dp(n, cached) == [(0, n, True)]


def test_single_word_gap_between_cached_stays_separate():
    # a 1-word gap wedged between two cached spans is its own span -- never
    # merged into a cached span (the merge only coalesces same-type neighbors).
    assert segment_dp(5, {(0, 2), (3, 5)}) == [
        (0, 2, True), (2, 3, False), (3, 5, True),
    ]


def test_tiling_is_gapless_and_non_overlapping():
    # every reconstruction must partition [0,n) exactly once.
    for n in range(0, 25):
        for cached in (set(), {(0, n)}, {(0, 1), (n - 1, n)} if n > 1 else set()):
            segs = segment_dp(n, cached)
            covered = sum(hi - lo for lo, hi, _ in segs)
            assert covered == n, f"n={n} cached={cached} -> {segs}"
            # spans are contiguous and ordered
            pos = 0
            for lo, hi, _ in segs:
                assert lo == pos and hi > lo
                pos = hi
            assert pos == n
