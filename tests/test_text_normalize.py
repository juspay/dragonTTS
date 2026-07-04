"""Tests for TTS number normalization (app/audio/text.py) and its wiring into
the cache key via CacheService._resolve."""

from __future__ import annotations

import pytest

from app.audio.text import (
    _expand_number,
    normalize_for_tts,
    normalize_numbers,
    num_to_words,
    prepend_leading_dot,
)
from app.cache.service import CacheService
from app.core.config import settings
from app.providers.base import AudioResult
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore


@pytest.fixture
async def svc(tmp_storage, fake_provider):
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    return CacheService(meta, blobs, lambda name: fake_provider if name == "cartesia" else None)


def _req(text: str) -> TTSRequest:
    return TTSRequest(
        model_id="cartesia:sonic-3.5", transcript=text, voice=CartesiaVoice(id="v1"),
        language="en", output_format=OutputFormat(), params={},
    )


# --- num_to_words (Indian grouping) ---------------------------------------


def test_num_to_words_indian():
    assert num_to_words(0) == "zero"
    assert num_to_words(5) == "five"
    assert num_to_words(99) == "ninety nine"
    assert num_to_words(599) == "five hundred ninety nine"
    assert num_to_words(100000) == "one lakh"          # the lakh requirement
    assert num_to_words(150000) == "one lakh fifty thousand"
    assert num_to_words(10000000) == "one crore"        # 8 digits still a number


# --- normalize_numbers ----------------------------------------------------


def test_hybrid_and_pure_collide():
    a = normalize_numbers("5 hundred 99 rupees का.")
    b = normalize_numbers("599 rupees का.")
    assert a == b == "five hundred ninety nine rupees का."


def test_decimals_and_long_ids():
    assert normalize_numbers("5.99 rupees") == "five point nine nine rupees"
    # 10 digits -> digit-by-digit (phone/order id)
    assert normalize_numbers("9876543210") == "nine eight seven six five four three two one zero"


def test_trailing_dot_is_not_a_decimal():
    # a sentence period after digits stays punctuation, not "point"
    assert normalize_numbers("99.") == "ninety nine."
    assert normalize_numbers("the price is 5.") == "the price is five."


def test_comma_grouping():
    assert normalize_numbers("1,00,000 rupees") == "one lakh rupees"
    assert normalize_numbers("1,500.99") == "one thousand five hundred point nine nine"


def test_leaves_words_and_trailing_punct():
    assert normalize_numbers("order किया है. 5 hundred 99 rupees का.") == \
        "order किया है. five hundred ninety nine rupees का."
    assert normalize_numbers("Pack of 2") == "Pack of two"


# --- normalize_for_tts: number normalization for the cache KEY (no dot) -----


def test_for_tts_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", False)
    assert normalize_for_tts("5 hundred 99", "elevenlabs") == "5 hundred 99"


def test_for_tts_is_number_only_never_dots(monkeypatch):
    """normalize_for_tts is KEY text: number expansion only. The ElevenLabs dot
    is a synth-only hint (prepend_leading_dot) and must NOT be baked into the
    key, else "your order" and ".your order" get separate entries."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    # elevenlabs: numbers expand, but NO leading dot
    assert normalize_for_tts("hello bro", "elevenlabs") == "hello bro"
    assert normalize_for_tts("5 hundred", "elevenlabs") == "five hundred"
    # an input leading "." on a number token is preserved by normalize_numbers,
    # but no EXTRA dot is ever prepended by normalize_for_tts
    assert normalize_for_tts(".5 hundred", "elevenlabs") == ".five hundred"


def test_for_tts_cartesia_is_number_only(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    assert normalize_for_tts("hello bro", "cartesia") == "hello bro"
    assert normalize_for_tts("5 hundred", "cartesia") == "five hundred"


# --- prepend_leading_dot: ElevenLabs SYNTH-only hint (never the key) --------


def test_prepend_dot_elevenlabs(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    assert prepend_leading_dot("hello bro", "elevenlabs") == ".hello bro"
    assert prepend_leading_dot("five hundred", "elevenlabs") == ".five hundred"
    # already starts with "." -> no double dot (idempotent)
    assert prepend_leading_dot(".five hundred", "elevenlabs") == ".five hundred"
    assert prepend_leading_dot(".abc", "elevenlabs") == ".abc"


def test_prepend_dot_off_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", False)
    assert prepend_leading_dot("hello bro", "elevenlabs") == "hello bro"
    assert prepend_leading_dot("five hundred", "elevenlabs") == "five hundred"


def test_prepend_dot_independent_of_normalize(monkeypatch):
    # the dot is independent of number expansion: it applies even when normalize
    # is off (so turning TTS_NORMALIZE_NUMBERS off never silently kills the dot).
    monkeypatch.setattr(settings, "tts_normalize_numbers", False)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    assert prepend_leading_dot("hello bro", "elevenlabs") == ".hello bro"


def test_prepend_dot_cartesia_unaffected(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    assert prepend_leading_dot("hello bro", "cartesia") == "hello bro"
    assert prepend_leading_dot("five hundred", "cartesia") == "five hundred"


def test_prepend_dot_empty_and_case_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    # must NOT turn empty/whitespace into a bare "."
    assert prepend_leading_dot("", "elevenlabs") == ""
    assert prepend_leading_dot("   ", "elevenlabs") == "   "
    # provider match is case- / whitespace-insensitive
    assert prepend_leading_dot("hello", "ElevenLabs") == ".hello"
    assert prepend_leading_dot("hello", " elevenlabs ") == ".hello"


# --- end-to-end: normalization collapses cache keys -----------------------


async def test_cache_key_collapses_599_and_5hundred99(svc, monkeypatch):
    """'5 hundred 99' and '599' normalize to one key -> second is a HIT."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    await svc.get_or_synthesize(_req("5 hundred 99 rupees"))
    cached, _rec, _p, _m, _key = await svc.check(_req("599 rupees"))
    assert cached


async def test_cache_key_collapses_comma_grouping(svc, monkeypatch):
    """'1,00,000' (comma form) and '100000' collapse to one key."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    await svc.get_or_synthesize(_req("1,00,000 rupees"))
    cached, _rec, _p, _m, _key = await svc.check(_req("100000 rupees"))
    assert cached


# --- review-driven hardening tests ---------------------------------------


def test_num_to_words_large_indian():
    # 8 digits (the max the expander treats as an Indian number): 9,99,99,999
    assert num_to_words(99999999) == "nine crore ninety nine lakh ninety nine thousand nine hundred ninety nine"
    # num_to_words itself still groups past one crore (10^8) when called directly
    assert num_to_words(100000000) == "ten crore"


def test_expand_number_defensive():
    # empty / punctuation-only -> unchanged (no crash, no malformed output)
    assert _expand_number("") == ""
    assert _expand_number(",,") == ""
    # trailing period with no decimal digits -> just the integer (no dangling "point")
    assert _expand_number("5.") == "five"


def test_normalize_for_tts_idempotent():
    """Feeds the cache key via every _resolve, so it MUST be idempotent."""
    cases = ["5 hundred 99 rupees", "599", ".5 hundred", "hello bro", "100000", ""]
    for text in cases:
        for provider in ("elevenlabs", "cartesia"):
            once = normalize_for_tts(text, provider)
            twice = normalize_for_tts(once, provider)
            assert once == twice, f"not idempotent: {text!r}/{provider} -> {once!r} -> {twice!r}"


def test_elevenlabs_empty_transcript_gets_no_bare_dot(monkeypatch):
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    # neither function may turn empty/whitespace into a bare "."
    assert normalize_for_tts("", "elevenlabs") == ""
    assert normalize_for_tts("   ", "elevenlabs") == ""
    assert prepend_leading_dot("", "elevenlabs") == ""
    assert prepend_leading_dot("   ", "elevenlabs") == "   "


def test_key_is_dot_free_regardless_of_provider_case(monkeypatch):
    """The cache key text must NEVER carry the ElevenLabs dot (it's synth-only),
    no matter how the provider name is cased."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    assert normalize_for_tts("hello", "ElevenLabs") == "hello"
    assert normalize_for_tts("hello", " elevenlabs ") == "hello"
    assert normalize_for_tts("5 hundred", "ELEVENLABS") == "five hundred"


class _RecordingProvider:
    """ElevenLabs stand-in that records the text it's asked to synthesize."""

    name = "elevenlabs"
    native_encoding = "pcm_s16le"
    native_sample_rate = 16000

    def __init__(self):
        self.seen: list[str] = []

    async def synth(self, *, text, voice_id, model, language, params) -> AudioResult:
        self.seen.append(text)
        return AudioResult(b"\x00\x01" * 400, "raw", "pcm_s16le", 16000)


async def test_elevenlabs_dot_prepends_through_service(tmp_storage, monkeypatch):
    """The leading-dot prepend flows from _resolve all the way to synth()."""
    monkeypatch.setattr(settings, "tts_normalize_numbers", True)
    monkeypatch.setattr(settings, "tts_leading_dot", True)
    prov = _RecordingProvider()
    meta = SQLiteMetadataStore(settings.db_path)
    await meta.init()
    blobs = FilesystemBlobStore(settings.blob_dir)
    await blobs.init()
    svc = CacheService(meta, blobs, lambda name: prov if name == "elevenlabs" else None)

    req = TTSRequest(
        model_id="elevenlabs:eleven_flash_v2_5", transcript="5 hundred 99 rupees",
        voice=CartesiaVoice(id="v1"), language="en", output_format=OutputFormat(), params={},
    )
    await svc.get_or_synthesize(req)
    assert prov.seen, "provider.synth was never called"
    # numbers expanded AND the leading dot prepended (ElevenLabs)
    assert prov.seen[0] == ".five hundred ninety nine rupees"
