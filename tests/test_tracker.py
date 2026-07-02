"""FrequencyTracker — predictive warming via timestamp-split.

On threshold cross, a phrase is synthed once with word timestamps and its audio
is split into every contiguous sub-phrase entry (substring closure). The cache
check-before-store means a sub-phrase already split out of a longer phrase is
never re-synthed.
"""

from __future__ import annotations

import pytest

from app.cache.service import CacheService
from app.cache.tracker import FrequencyTracker
from app.core.config import settings
from app.providers.base import AudioResult
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


def _req(text: str) -> TTSRequest:
    return TTSRequest(
        model_id="cartesia:sonic-3.5",
        transcript=text,
        voice=CartesiaVoice(id="v1"),
        language="en",
        output_format=OutputFormat(),
        params={},
    )


class _FakeTimestampProvider:
    """Fake cartesia-like provider: synth + synth_with_timestamps.

    Timestamped synth returns aligned words + 0.5s/word audio so warm_split can
    slice sub-phrases. Counts timestamped synths in ``ts_calls``.
    """

    name = "cartesia"
    native_encoding = "pcm_s16le"
    native_sample_rate = 16000

    def __init__(self):
        self.calls = 0
        self.ts_calls = 0

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        self.calls += 1
        words = text.split()
        return AudioResult(b"\x00\x01" * (8000 * len(words)), "raw", "pcm_s16le", 16000)

    async def synth_with_timestamps(self, *, text, voice_id, model, language, params):
        self.ts_calls += 1
        words = text.split()
        audio = b"\x00\x01" * (8000 * len(words))  # 0.5s/word @16k mono
        starts = [0.5 * i for i in range(len(words))]
        return audio, words, starts


@pytest.fixture
async def svc(tmp_storage, fake_provider):
    """CacheService backed by the no-timestamp FakeProvider (exercises fallback)."""
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None)


@pytest.fixture
async def ts_svc(tmp_storage):
    """CacheService backed by _FakeTimestampProvider (exercises the split path)."""
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    prov = _FakeTimestampProvider()
    svc = CacheService(meta, blobs, lambda name: prov if name == "cartesia" else None)
    svc._fake = prov  # type: ignore[attr-defined]
    return svc


def _tracker_for(svc, monkeypatch):
    monkeypatch.setattr(settings, "predictive_warm_enabled", True)
    monkeypatch.setattr(settings, "predictive_warm_threshold", 3.0)
    monkeypatch.setattr(settings, "predictive_warm_threshold_step", 0.0)  # flat
    monkeypatch.setattr(settings, "predictive_warm_threshold_floor", 1.0)
    monkeypatch.setattr(settings, "predictive_warm_min_words", 1)
    monkeypatch.setattr(settings, "predictive_warm_max_words", 6)
    t = FrequencyTracker(svc)
    svc.attach_tracker(t)
    return t


async def _cached(svc: CacheService, text: str) -> bool:
    cached, *_ = await svc.check(_req(text))
    return cached


# -- split path (timestamp provider) --------------------------------------


async def test_warm_split_creates_all_subphrases(ts_svc, monkeypatch):
    tracker = _tracker_for(ts_svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="how are you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    # Every contiguous sub-phrase is cached from the single synth + split.
    assert await _cached(ts_svc, "how are you")
    assert await _cached(ts_svc, "how are")
    assert await _cached(ts_svc, "are you")
    assert await _cached(ts_svc, "how")
    assert await _cached(ts_svc, "are")
    assert await _cached(ts_svc, "you")
    assert ts_svc._fake.ts_calls == 1  # one timestamped synth, not k²


async def test_warm_split_skips_subphrase_already_split(ts_svc, monkeypatch):
    tracker = _tracker_for(ts_svc, monkeypatch)
    # "how are you" warms + splits -> "how are" is already cached.
    for _ in range(3):
        await tracker.observe(
            text="how are you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert ts_svc._fake.ts_calls == 1
    # Now "how are" crosses threshold on its own — already cached, no new synth.
    for _ in range(3):
        await tracker.observe(
            text="how are", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert ts_svc._fake.ts_calls == 1  # check-before-synth skipped it


# -- fallback path (no-timestamp provider) --------------------------------


async def test_warm_fallback_stores_full_phrase_only(svc, fake_provider, monkeypatch):
    tracker = _tracker_for(svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="how are you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(svc, "how are you")  # full phrase stored
    assert not await _cached(svc, "how are")  # no timestamps -> no split
    assert not await _cached(svc, "how")
    assert fake_provider.calls == 1


async def test_below_threshold_does_not_warm(svc, fake_provider, monkeypatch):
    tracker = _tracker_for(svc, monkeypatch)
    for _ in range(2):  # threshold is 3
        await tracker.observe(
            text="good morning", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert not await _cached(svc, "good morning")
    assert fake_provider.calls == 0


async def test_already_warmed_not_re_synthesized(svc, fake_provider, monkeypatch):
    tracker = _tracker_for(svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="thank you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert fake_provider.calls == 1
    for _ in range(5):
        await tracker.observe(
            text="thank you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert fake_provider.calls == 1


async def test_separates_clauses_and_warms_only_recurring_one(svc, fake_provider, monkeypatch):
    tracker = _tracker_for(svc, monkeypatch)
    for word in ("apple", "banana", "cherry"):
        await tracker.observe(
            text=f"have a nice day. {word}", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(svc, "have a nice day.")  # recurs -> warmed (period kept)
    assert not await _cached(svc, "apple")  # one-off -> not warmed
    assert fake_provider.calls == 1


async def test_period_splits_segments_no_cross_boundary(ts_svc, monkeypatch):
    """A '.' ends a segment: each side is warmed independently (period kept on
    the boundary token) and a cross-'.' fragment is never cached."""
    tracker = _tracker_for(ts_svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="hi there. how are you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(ts_svc, "hi there.")        # segment 1 (period kept)
    assert await _cached(ts_svc, "how are you")      # segment 2
    assert not await _cached(ts_svc, "there. how")   # cross-'.' -> never tracked
    assert not await _cached(ts_svc, "hi there. how are you")  # spans the boundary


async def test_warm_keeps_punctuation_matching_live_keys(ts_svc, monkeypatch):
    """Warmed keys keep punctuation so they match a live request for the same
    text. (The old clause-split stripped it, so warmed entries never hit.)"""
    tracker = _tracker_for(ts_svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="how are you.", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(ts_svc, "how are you.")     # period kept -> matches live key
    assert not await _cached(ts_svc, "how are you")  # no-period form is a different key


async def test_split_chars_env_controls_delimiters(ts_svc, monkeypatch):
    """PREDICTIVE_WARM_SPLIT_CHARS picks the delimiters; '.?!' also splits on '?'
    (the default '.' would leave '?' continuous)."""
    monkeypatch.setattr(settings, "predictive_warm_split_chars", ".?!")
    tracker = _tracker_for(ts_svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="hi there? how are you", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(ts_svc, "hi there?")        # '?' ended the segment
    assert await _cached(ts_svc, "how are you")
    assert not await _cached(ts_svc, "there? how")   # cross-'?' not cached


async def test_decay_ages_out_stale_phrases(svc, fake_provider, monkeypatch):
    tracker = _tracker_for(svc, monkeypatch)
    monkeypatch.setattr(settings, "predictive_warm_decay_factor", 0.5)
    monkeypatch.setattr(settings, "predictive_warm_min_floor", 0.6)
    for _ in range(2):  # below threshold
        await tracker.observe(
            text="fading phrase", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    state = next(iter(tracker._contexts.values()))
    assert state.counter[("fading", "phrase")] == 2.0
    tracker._decay()  # -> 1.0
    assert state.counter[("fading", "phrase")] == 1.0
    tracker._decay()  # -> 0.5 (< 0.6 floor -> pruned)
    assert ("fading", "phrase") not in state.counter


async def test_context_separation(ts_svc, monkeypatch):
    tracker = _tracker_for(ts_svc, monkeypatch)
    for _ in range(3):
        await tracker.observe(
            text="same words", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(ts_svc, "same words")
    assert len(tracker._contexts) == 1


# -- length-scaled threshold ----------------------------------------------


async def test_threshold_scales_with_phrase_length(ts_svc, monkeypatch):
    """Long phrases warm after fewer occurrences than short ones.

    base=3.0, step=0.5 -> a 4-word phrase's threshold is 3.0 - 3*0.5 = 1.5, so 2
    occurrences warm it; a 1-word phrase's threshold stays 3.0, so 2 do not.
    Same occurrence count, opposite outcome — purely by phrase length.
    """
    tracker = _tracker_for(ts_svc, monkeypatch)
    monkeypatch.setattr(settings, "predictive_warm_threshold_step", 0.5)

    # 4-word phrase warms after only 2 occurrences (threshold 1.5).
    for _ in range(2):
        await tracker.observe(
            text="alpha beta gamma delta", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert await _cached(ts_svc, "alpha beta gamma delta")
    assert ts_svc._fake.ts_calls == 1  # one synth; sub-phrases split out, none re-synthed

    # 1-word phrase does NOT warm after 2 occurrences (threshold 3.0).
    for _ in range(2):
        await tracker.observe(
            text="kappa", provider="cartesia", voice_id="v1",
            model="sonic-3.5", language="en", params={},
        )
    await tracker.drain()
    assert not await _cached(ts_svc, "kappa")
