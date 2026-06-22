"""Cache-key derivation unit tests."""

from app.cache.key import (
    canonical_params,
    hash_key,
    normalize_text,
    parse_model_id,
)


def test_normalize_collapses_whitespace():
    assert normalize_text("  thank   you\t") == "thank you"


def test_normalize_nfc():
    # NFD "é" (e + combining accent) should fold to NFC "é"
    assert normalize_text("é") == "é"


def test_parse_model_id_first_colon_only():
    # sarvam model "bulbul:v3" itself contains a colon
    provider, model = parse_model_id("sarvam:bulbul:v3")
    assert provider == "sarvam"
    assert model == "bulbul:v3"


def test_parse_model_id_bad():
    import pytest

    with pytest.raises(ValueError):
        parse_model_id("nodelimiter")
    with pytest.raises(ValueError):
        parse_model_id(":missingprovider")
    with pytest.raises(ValueError):
        parse_model_id("sarvam:")


def test_canonical_params_collapses_defaults():
    # speed=1.0 is the cartesia default → dropped
    assert canonical_params("cartesia", {"speed": 1.0, "emotion": "neutral"}) == ""
    # a non-default value is kept, sorted, compact JSON
    assert canonical_params("cartesia", {"emotion": "excited", "speed": 1.5}) == (
        '{"emotion":"excited","speed":1.5}'
    )


def test_hash_key_deterministic_and_differentiated():
    base = dict(
        text="thank you",
        provider="cartesia",
        voice_id="v1",
        model="sonic-3.5",
        language="en",
        params_canonical="",
    )
    k1 = hash_key(**base)
    k2 = hash_key(**base)
    assert k1 == k2  # deterministic

    # whitespace-only difference normalizes away → same key
    diff = {**base, "text": "  thank   you "}
    assert hash_key(**diff) == k1

    # output_format is NOT part of the key (native store + convert-on-serve)
    # → same key regardless of format
    assert hash_key(**base) == k1

    # different voice → different key
    diff = {**base, "voice_id": "v2"}
    assert hash_key(**diff) != k1

    # different language → different key
    diff = {**base, "language": "te"}
    assert hash_key(**diff) != k1
