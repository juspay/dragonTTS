"""TTS text normalization — expand numbers to words so providers read amounts
correctly (ElevenLabs garbles digit+word hybrids like "5 hundred 99").

Applied in :meth:`CacheService._resolve` after :func:`normalize_text`, to BOTH
the text sent to the provider and the cache key (so "599" and "5 hundred 99"
collapse to one entry).

Scope (gated by ``TTS_NORMALIZE_NUMBERS``):
  * <= 8 integer digits -> Indian cardinal (lakh/crore): 100000 -> "one lakh",
    10000000 -> "one crore"
  * >= 9 integer digits -> digit-by-digit (phone/order ids)
  * decimals -> "point" + up to 10 places: 5.99 -> "five point nine nine"
  * word-multipliers (hundred/thousand/lakh/crore) left as-is

Leading-dot handling (``TTS_LEADING_DOT_MODE``, ElevenLabs only — Cartesia reads
a stray "." fine): a "." immediately before the first digit (attached ".5" or
spaced ". 5") is resolved per mode: drop / space / nospace.
"""

from __future__ import annotations

import re

from app.core.config import settings

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# A number token: digits + commas, optionally a decimal point WITH digits after.
# A trailing "." (sentence period, no following digits) is NOT part of the number
# (else "99." would parse as a decimal -> "ninety nine point ").
_TOKEN_NUM = re.compile(r"^(\W*)(\d[\d,]*(?:\.\d+)?)(\W*)$")


def _under_thousand(x: int) -> str:
    parts = []
    if x >= 100:
        parts.append(f"{_ONES[x // 100]} hundred"); x %= 100
    if x:
        parts.append(_ONES[x] if x < 20 else _TENS[x // 10] + ("" if x % 10 == 0 else " " + _ONES[x % 10]))
    return " ".join(parts) or "zero"


def num_to_words(n: int) -> str:
    """Indian cardinal: lakh (10^5), crore (10^7). 100000 -> 'one lakh'."""
    if n == 0:
        return "zero"
    parts = []
    cr, n = divmod(n, 10_000_000)
    la, n = divmod(n, 100_000)
    th, n = divmod(n, 1_000)
    if cr:
        parts.append(_under_thousand(cr) + " crore")
    if la:
        parts.append(_under_thousand(la) + " lakh")
    if th:
        parts.append(_under_thousand(th) + " thousand")
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def _expand_number(s: str) -> str:
    """Expand a digit string (commas / one decimal allowed) to words. Defensive
    against empty / punctuation-only input (returns it unchanged)."""
    s = s.replace(",", "").strip()
    if not s or not any(c.isdigit() for c in s):
        return s
    if "." in s:  # decimal: integer part as words + "point" + each digit (<=10)
        whole, _, frac = s.partition(".")
        digits = [d for d in frac if d.isdigit()][:10]
        int_part = num_to_words(int(whole)) if whole.isdigit() else "zero"
        if not digits:               # "5." (trailing period) -> just the integer
            return int_part
        return int_part + " point " + " ".join(_ONES[int(d)] for d in digits)
    if len(s) >= 9:  # 9+ integer digits -> digit-by-digit (phone/order id)
        return " ".join(_ONES[int(d)] for d in s if d.isdigit())
    return num_to_words(int(s))


def normalize_numbers(text: str) -> str:
    """Expand standalone digit tokens to English words (Indian grouping).

    Trailing sentence punctuation is kept ("99." -> "ninety nine."). A leading
    "." before a digit stays on the token (".5" -> ".five"); the ElevenLabs
    leading-dot prepend is handled by :func:`normalize_for_tts`."""
    out = []
    for tok in text.split():
        m = _TOKEN_NUM.match(tok)
        if not m:
            out.append(tok); continue
        out.append(m.group(1) + _expand_number(m.group(2)) + m.group(3))
    return " ".join(out)


def normalize_for_tts(text: str, provider: str) -> str:
    """Number expansion (all providers) + ElevenLabs leading-dot prepend.

    When ``tts_leading_dot`` is true and the provider is ElevenLabs, ensure the
    text starts with "." — prepend one (no space) if missing, leave it if already
    present. Other providers are unaffected. No-op when
    ``TTS_NORMALIZE_NUMBERS`` is false."""
    if not settings.tts_normalize_numbers:
        return text
    text = normalize_numbers(text)
    if (
        provider
        and provider.strip().lower() == "elevenlabs"
        and settings.tts_leading_dot
        and text
        and not text.startswith(".")
    ):
        text = "." + text
    return text
