"""Application configuration — env-backed settings + provider defaults.

Provider API-key env names mirror clairvoyance so the same k8s Secret can be
reused across both services.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Per-provider defaults. Ported from clairvoyance BB_SPEECH_PROVIDER_DEFAULTS.
# Used as fallbacks when a request omits a field, and as the canonicalization
# baseline for cache-key collapsing (see app/cache/key.py).
PROVIDER_DEFAULTS: dict[str, dict] = {
    "cartesia": {
        "voice_id": "bec003e2-3cb3-429c-8468-206a393c67ad",
        "model": "sonic-3.5",
        "speed": 1.0,
        "volume": 1.0,
        "emotion": "neutral",
        "language": "en",
    },
    "sarvam": {
        "voice_id": "shreya",
        "model": "bulbul:v3",
        "language": "en-IN",
        "speed": 0.9,
        "pitch": 0.0,
    },
    "elevenlabs": {
        "voice_id": "fG9s0SXJb213f4UxVHyG",
        "model": "eleven_flash_v2_5",
        "speed": 1.15,
        "language": "en",
    },
    "gemini": {
        "voice_id": "Kore",
        "model": "gemini-3.1-flash-tts-preview",
        "language": "en-IN",
    },
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_DEFAULTS.keys())

# Indian-residency ElevenLabs host (matches clairvoyance's residency endpoint).
# The residency KEY is the only thing that must be supplied; the URL defaults
# here and is coerced non-empty by the validator below even if .env blanks it.
ELEVENLABS_INDIAN_RESIDENCY_URL = "https://api.in.residency.elevenlabs.io"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Provider credentials (env names mirror clairvoyance) ---
    cartesia_api_key: str = ""
    sarvam_api_key: str = ""
    # ElevenLabs runs ONLY against the Indian-residency endpoint (the only creds
    # with API access). Env names mirror clairvoyance's residency Secret keys so
    # the same k8s Secret is reused; the single "elevenlabs" route resolves here.
    elevenlabs_indian_residency_api_key: str = ""
    elevenlabs_indian_residency_base_url: str = ELEVENLABS_INDIAN_RESIDENCY_URL

    @field_validator("elevenlabs_indian_residency_base_url", mode="after")
    @classmethod
    def _coerce_residency_url(cls, v: str) -> str:
        # An empty ELEVENLABS_INDIAN_RESIDENCY_BASE_URL= line in .env would
        # otherwise blank the default; fall back to the known residency host.
        return v or ELEVENLABS_INDIAN_RESIDENCY_URL
    google_credentials_json: str = ""
    google_credentials_path: str = ""

    # --- Storage ---
    db_path: str = "data/dragontts.db"
    blob_dir: str = "data/blobs"

    # --- Cache policy ---
    max_cache_bytes: int = 0  # 0 = unlimited
    ttl_seconds: int = 0  # 0 = no expiry
    enable_write_through: bool = True

    # --- Performance ---
    thread_pool_workers: int = 32  # asyncio.to_thread pool size
    bulk_create_max: int = 1000  # hard cap on /tts/create/bulk items
    # Number of warm, persistent Cartesia streaming sockets kept ready for cache
    # misses (each multiplexes many utterances by context_id). Set via env, e.g.
    # CARTESIA_STREAM_POOL_SIZE=4. 0 => open a fresh socket per miss (no pooling).
    cartesia_stream_pool_size: int = 2
    # Warm ElevenLabs multi-context WS sockets, pooled PER VOICE (the voice is in
    # the WS URL). Each socket multiplexes up to 5 concurrent contexts. Streaming
    # misses reuse a warm socket; if none is ready, fall back to one-shot HTTP.
    elevenlabs_stream_pool_size: int = 2
    # Max server-silence gap (seconds) after audio starts that ends an ElevenLabs
    # WS utterance (ElevenLabs delays is_final ~20s). Lower = faster stream
    # close/turn-end; raise if long utterances ever truncate at a >N s pause.
    elevenlabs_stream_idle_timeout: float = 0.8
    # Warm Sarvam WS sockets. Sarvam is NOT multiplexed (one utterance per socket
    # at a time), so this is a LIFO stack of warm, pre-configured connections. 0
    # => stream via a fresh socket per miss (no pooling).
    sarvam_stream_pool_size: int = 2
    # Force IPv4 for the Cartesia WS handshake. Some networks advertise IPv6
    # (AAAA) for api.cartesia.ai but black-hole the SYN, hanging the handshake
    # while IPv4 works fine. Safe default (IPv4 always reaches Cartesia).
    cartesia_ws_force_ipv4: bool = True
    # --- Resilience: per-provider bulkhead (concurrency cap) + rate limit ---
    # Caps how many in-flight synth/stream calls ONE provider may have so a slow
    # or hung provider can't exhaust shared resources (event loop, worker pool,
    # memory) and starve the others. Global defaults; override per provider via
    # the JSON env PROVIDER_RESILIENCE='{"cartesia":{"max_concurrent":20,
    # "rate_per_sec":8,"wait_timeout_ms":2000}, ...}'. A 0 value disables that
    # limiter. The bulkhead waits up to wait_timeout_ms for a slot, else 503.
    provider_max_concurrent_synths: int = 24
    provider_rate_limit_per_sec: float = 0.0
    provider_bulkhead_wait_timeout_ms: int = 2500
    provider_resilience_overrides: dict = Field(default_factory=dict)
    # --- Write-behind metrics (off the hot HIT path) ---
    # HIT touch/metric updates are batched + flushed by a background task so a HIT
    # returns audio without awaiting a SQLite write. Flush by interval or batch
    # size, and on graceful shutdown. enabled=false -> synchronous writes.
    metrics_write_behind_enabled: bool = True
    metrics_flush_interval_ms: int = 500
    metrics_flush_batch_size: int = 64
    # --- Predictive cache warming (Part 1: frequency-based auto-warm) ---
    # Tracks recurring phrase substrings across requests and warms the frequent
    # ones into the cache so Part 2 (segment + stitch) can assemble them.
    predictive_warm_enabled: bool = True
    # Length-scaled warm threshold. A short phrase (e.g. "hi") is a substring of
    # many longer ones, so its decayed count is the sum of all of them — it
    # would dominate warming and cache trivial fragments. Longer phrases are the
    # valuable scripted lines but occur less often. So the threshold STARTS HIGH
    # for the shortest tracked phrase and steps DOWN per extra word, to a floor:
    #   threshold(len) = max(threshold - (len - min_words) * step, floor)
    # e.g. threshold=3.0, step=0.5, floor=1.5, min_words=1 ->
    #     1 word=3.0, 2=2.5, 3=2.0, 4=1.5, 5..6=1.5 (floored).
    # Set step=0.0 for a flat threshold (legacy behaviour).
    predictive_warm_threshold: float = 3.0  # base threshold (shortest phrase)
    predictive_warm_threshold_step: float = 0.5  # subtracted per extra word
    predictive_warm_threshold_floor: float = 1.5  # never below this
    predictive_warm_min_words: int = 1  # shortest phrase length to track
    predictive_warm_max_words: int = 6  # cap on contiguous-substring length
    predictive_warm_decay_factor: float = 0.94  # counts x this each interval; with the 5-min interval below this is a ~1h half-life
    predictive_warm_decay_interval_s: int = 300  # how often decay runs (5 min)
    predictive_warm_min_floor: float = 0.5  # prune counts below this after decay
    # --- Predictive stitching (Part 2: serve a MISS from cached sub-phrases) ---
    # On a full-text MISS, binary-search cached prefix/suffix, synth only the
    # gaps, cross-fade at seams. Skipped below the coverage gate.
    predictive_stitch_enabled: bool = True
    # Stitch a MISS from cached sub-phrases when at least this fraction is cached
    # (miss <= 1 - this). Default 0.25 = latency-favored (stitch when miss < 75%):
    # the gap synth is <=75% of the phrase so it still beats a full synth on the
    # first request, and the assembled clip is cached for instant repeat HITs.
    # Override per-env via PREDICTIVE_STITCH_MIN_COVERAGE — raise to 0.5 for
    # quality-favored (fewer seams); don't lower below ~0.2 (assembly overhead
    # then outweighs the synth savings).
    predictive_stitch_min_coverage: float = 0.25
    # Apply stitch on the /tts/stream path too (assemble + stream + cache).
    # Independent of the one-shot flag: live streaming has lower TTFB, so this is
    # the trade of "first request waits for gap-synth + assembly" vs "reuse cached
    # sub-phrases and cache the assembled clip for instant repeat HITs".
    predictive_stitch_stream_enabled: bool = True
    # --- Stitch seam-DSP knobs (numpy) — tune assembled-clip quality via env. ---
    predictive_stitch_xfade_ms: float = 15.0        # crossfade overlap at each splice (10-25ms; short clicks, long smears)
    predictive_stitch_target_rms_db: float = -20.0  # per-fragment loudness target (speech ~-23..-18 dBFS)
    predictive_stitch_rms_floor_db: float = -55.0   # below this a fragment isn't amplified (don't hiss up a breath/gap)
    predictive_stitch_sil_relative_db: float = 25.0 # silence gate: a window this many dB below the clip peak is trimmed (HIGHER = more aggressive gap cutting)
    predictive_stitch_sil_guard_ms: float = 3.0     # sliver kept at each trimmed edge so onsets/offsets survive
    predictive_stitch_zc_search_ms: float = 4.0     # window scanned for a zero crossing to anchor each splice (avoids clicks)

    @property
    def configured_providers(self) -> list[str]:
        """Providers whose required credential is present at startup."""
        live: list[str] = []
        if self.cartesia_api_key:
            live.append("cartesia")
        if self.sarvam_api_key:
            live.append("sarvam")
        if self.elevenlabs_indian_residency_api_key:
            live.append("elevenlabs")
        if self.google_credentials_json or self.google_credentials_path:
            live.append("gemini")
        return live


settings = Settings()
