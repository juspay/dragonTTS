"""TTS normalization A/B probe.

Generates each phrase BOTH ways through dragonTTS's real text pipeline:
  RAW  = normalize_text only            (the old path, before number normalization)
  NORM = normalize_text + normalize_for_tts   (our new path)
across Cartesia and ElevenLabs. Listen to RAW vs NORM for each phrase/provider.

Run from repo root:  .venv/bin/python scripts/raw_elevenlabs_probe.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.text import normalize_for_tts  # noqa: E402
from app.cache.key import normalize_text  # noqa: E402
from app.providers.registry import ProviderRegistry  # noqa: E402

OUT_DIR = Path("/tmp")

# Diverse number/word mixes ("all kinds") + non-number "." cases (ElevenLabs safety).
PHRASES = [
    ("hybrid",   "5 hundred 99 rupees का."),        # digit+word hybrid (the prod bug)
    ("pure",     "599 rupees का."),                 # pure digits
    ("lakh",     "100000 rupees का."),              # lakh
    ("big_lakh", "1500000 rupees"),                 # 15 lakh
    ("decimal",  "5.99 rupees"),                    # decimal
    ("phone",    "call 9876543210"),                # 10-digit -> digit-by-digit
    ("mixed",    "आपका order 5 hundred 99 rupees का है."),  # Hinglish sentence w/ amount
    ("dot_word", ". hello there"),                  # leading dot before a WORD (non-number)
    ("mid_dot",  "okay. how are you"),              # mid-sentence dot
    ("end_dot",  "thanks."),                        # trailing dot
]

PROVIDER_CONFIG = {
    "cartesia": ("", ""),
    "elevenlabs": ("iB2rIwm9cQCRGWoKDRtX", "eleven_flash_v2_5"),
}


def pcm_to_mp3(pcm: bytes, out: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
        f.write(pcm); raw = Path(f.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "16000",
             "-ac", "1", "-i", str(raw), str(out)], check=True)
    finally:
        raw.unlink(missing_ok=True)


async def main() -> int:
    registry = ProviderRegistry(); registry.build()
    print(f"configured providers: {registry.configured()}\n")
    providers = {n: registry.get(n) for n in PROVIDER_CONFIG if registry.get(n)}

    for pname, prov in providers.items():
        voice, model = PROVIDER_CONFIG[pname]
        print(f"== {pname} ==")
        for label, phrase in PHRASES:
            raw_text = normalize_text(phrase)
            norm_text = normalize_for_tts(raw_text, pname)
            for method, text in (("RAW", raw_text), ("NORM", norm_text)):
                try:
                    result = await prov.synth(text=text, voice_id=voice, model=model,
                                              language=None, params={})
                except Exception as e:
                    print(f"  [{label:9}/{method}] ERROR: {e}"); continue
                out = OUT_DIR / f"our_{pname}_{label}_{method}.mp3"
                pcm_to_mp3(result.audio, out)
                tag = "" if method == "RAW" or text == raw_text else f"  <- {text!r}"
                print(f"  [{label:9}/{method:4}] {out.name} ({out.stat().st_size:,} B){tag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
