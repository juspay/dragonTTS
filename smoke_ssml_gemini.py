"""Focused live test: ElevenLabs SSML(on) + Gemini — the two paths smoke_live skips.

1. ElevenLabs SSML ON vs OFF for the SAME text: ON must be ~1.5s longer (the
   <break/> is parsed into a real pause, not read aloud). Validates both synth()
   and stream_synth() with SSML, and confirms the stream-routed _synthesize cache
   path still yields whole-frame, non-silent pcm_s16le@16k.
2. Cache-key isolation: SSML-on vs SSML-off for identical text derive DIFFERENT
   keys (so an SSML HIT never serves a non-SSML render).
3. Gemini synth + stream_synth happy path (the #9 bug only bites on gRPC failure,
   so this confirms normal operation still works).

Run:  uv run python smoke_ssml_gemini.py
"""

from __future__ import annotations

import asyncio

import numpy as np

from app.cache.service import CacheService
from app.core.config import PROVIDER_DEFAULTS, settings
from app.providers.elevenlabs import ElevenLabsProvider
from app.providers.gemini import GeminiProvider
from app.schemas.tts import CartesiaVoice, OutputFormat, TTSRequest
from app.storage.filesystem import FilesystemBlobStore
from app.storage.sqlite import SQLiteMetadataStore

TEXT_OFF = "Hello. Then a pause. Goodbye."
TEXT_ON = 'Hello. <break time="1.5s"/> Then a pause. Goodbye.'


def _stats(audio: bytes) -> tuple[int, float, float]:
    """(n_bytes, dur_ms, peak) for pcm_s16le@16k."""
    n = len(audio)
    if n == 0 or n % 2 != 0:
        return n, 0.0, 0.0
    s = np.frombuffer(audio, dtype="<i2").astype(np.int32)
    return n, n / 2 / 16000 * 1000, float(np.max(np.abs(s))) if len(s) else 0.0


async def drain(gen):
    out = bytearray()
    async for c in gen:
        out += c
    return bytes(out)


async def test_ssml(prov: ElevenLabsProvider) -> dict:
    # synth: OFF vs ON
    ar_off = await prov.synth(text=TEXT_OFF, voice_id=None, model=None,
                              language=None, params={})
    ar_on = await prov.synth(text=TEXT_ON, voice_id=None, model=None,
                             language=None, params={"enable_ssml_parsing": True})
    _, dur_off_ms, peak_off = _stats(ar_off.audio)
    _, dur_on_ms, peak_on = _stats(ar_on.audio)

    # stream: ON (exercises stream_synth WS path with SSML)
    streamed = await drain(prov.stream_synth(
        text=TEXT_ON, voice_id=None, model=None, language=None,
        params={"enable_ssml_parsing": True}))
    _, dur_stream_ms, peak_stream = _stats(streamed)

    delta_ms = dur_on_ms - dur_off_ms
    return {
        "synth_off_ms": round(dur_off_ms), "synth_off_peak": int(peak_off),
        "synth_on_ms": round(dur_on_ms), "synth_on_peak": int(peak_on),
        "stream_on_ms": round(dur_stream_ms), "stream_on_peak": int(peak_stream),
        "break_delta_ms": round(delta_ms),
        "ssml_parsed": delta_ms >= 1000,  # ~1.5s break => >=1s longer
        "synth_on_ok": dur_on_ms > 0 and peak_on > 50,
        "stream_on_ok": dur_stream_ms > 0 and peak_stream > 50,
    }


async def test_cache_key_isolation() -> dict:
    """Same text, SSML on vs off -> two distinct cache keys (two MISSes)."""
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="dragontts-ssml-"))
    meta = SQLiteMetadataStore(str(tmp / "k.db"))
    await meta.init()
    blobs = FilesystemBlobStore(str(tmp / "kb"))
    await blobs.init()
    prov = ElevenLabsProvider()

    def get_provider(n, _p=prov):
        return _p if n == "elevenlabs" else None

    svc = CacheService(meta, blobs, get_provider)

    def req(params):
        return TTSRequest(
            model_id=f"elevenlabs:{PROVIDER_DEFAULTS['elevenlabs']['model']}",
            transcript="Cache isolation check sentence.",
            voice=CartesiaVoice(id=PROVIDER_DEFAULTS["elevenlabs"]["voice_id"]),
            language=PROVIDER_DEFAULTS["elevenlabs"]["language"],
            output_format=OutputFormat(),
            params=params,
        )

    _, h_off = await svc.get_or_synthesize(req({}))
    _, h_on = await svc.get_or_synthesize(req({"enable_ssml_parsing": True}))
    await prov.aclose()
    return {
        "off_status": h_off["X-Cache"],
        "on_status": h_on["X-Cache"],
        "isolated": h_off["X-Cache"] == "MISS" and h_on["X-Cache"] == "MISS",
    }


async def test_gemini(prov: GeminiProvider) -> dict:
    ar = await prov.synth(text="Gemini live synthesis check.",
                          voice_id=None, model=None, language=None, params={})
    _, dur_synth_ms, peak_s = _stats(ar.audio)
    streamed = await drain(prov.stream_synth(
        text="Gemini live stream check.", voice_id=None, model=None,
        language=None, params={}))
    _, dur_stream_ms, peak_st = _stats(streamed)
    return {
        "synth_ms": round(dur_synth_ms), "synth_peak": int(peak_s),
        "synth_ok": dur_synth_ms > 0 and peak_s > 50,
        "stream_ms": round(dur_stream_ms), "stream_peak": int(peak_st),
        "stream_ok": dur_stream_ms > 0 and peak_st > 50,
    }


def mark(ok):
    return "PASS" if ok else "FAIL"


async def main():
    print("=" * 72)
    print("SSML + Gemini focused live test")
    print("=" * 72)

    el = ElevenLabsProvider()
    await el.warm()
    print("\n[1] ElevenLabs SSML on vs off (1.5s break)")
    r = await test_ssml(el)
    await el.aclose()
    print(f"    synth OFF     : {r['synth_off_ms']}ms, peak={r['synth_off_peak']}")
    print(f"    synth ON      : {r['synth_on_ms']}ms, peak={r['synth_on_peak']}  "
          f"[{mark(r['synth_on_ok'])}]")
    print(f"    stream ON     : {r['stream_on_ms']}ms, peak={r['stream_on_peak']}  "
          f"[{mark(r['stream_on_ok'])}]")
    print(f"    break effect  : +{r['break_delta_ms']}ms  (>=1000 = SSML parsed)  "
          f"[{mark(r['ssml_parsed'])}]")

    print("\n[2] Cache-key isolation (SSML on vs off, same text)")
    iso = await test_cache_key_isolation()
    print(f"    OFF -> {iso['off_status']} | ON -> {iso['on_status']}  "
          f"[{mark(iso['isolated'])}]")

    print("\n[3] Gemini synth + stream (happy path)")
    if settings.google_credentials_json or settings.google_credentials_path:
        gem = GeminiProvider()
        g = await test_gemini(gem)
        await gem.aclose()
        print(f"    synth  : {g['synth_ms']}ms, peak={g['synth_peak']}  "
              f"[{mark(g['synth_ok'])}]")
        print(f"    stream : {g['stream_ms']}ms, peak={g['stream_peak']}  "
              f"[{mark(g['stream_ok'])}]")
        gem_ok = g["synth_ok"] and g["stream_ok"]
    else:
        print("    (Gemini not configured — skipped)")
        gem_ok = True

    print("\n" + "=" * 72)
    all_ok = (r["synth_on_ok"] and r["stream_on_ok"] and r["ssml_parsed"]
              and iso["isolated"] and gem_ok)
    print("RESULT:", "ALL PASS ✓" if all_ok else "FAILURES (see above)")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
