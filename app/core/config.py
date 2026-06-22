"""Application configuration — env-backed settings + provider defaults.

Provider API-key env names mirror clairvoyance so the same k8s Secret can be
reused across both services.
"""

from __future__ import annotations

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
    "elevenlabs-in": {
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Provider credentials (env names mirror clairvoyance) ---
    cartesia_api_key: str = ""
    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_indian_residency_api_key: str = ""
    elevenlabs_indian_residency_base_url: str = ""
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
    blob_cache_bytes: int = 64 * 1024 * 1024  # in-memory LRU cap for hot blobs
    thread_pool_workers: int = 64  # asyncio.to_thread pool size
    bulk_create_max: int = 1000  # hard cap on /tts/create/bulk items
    # Number of warm, persistent Cartesia streaming sockets kept ready for cache
    # misses (each multiplexes many utterances by context_id). Set via env, e.g.
    # CARTESIA_STREAM_POOL_SIZE=4. 0 => open a fresh socket per miss (no pooling).
    cartesia_stream_pool_size: int = 2
    # Force IPv4 for the Cartesia WS handshake. Some networks advertise IPv6
    # (AAAA) for api.cartesia.ai but black-hole the SYN, hanging the handshake
    # while IPv4 works fine. Safe default (IPv4 always reaches Cartesia).
    cartesia_ws_force_ipv4: bool = True
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
    predictive_warm_decay_factor: float = 0.9  # multiply all counts each interval
    predictive_warm_decay_interval_s: int = 60  # how often decay runs
    predictive_warm_min_floor: float = 0.5  # prune counts below this after decay
    # --- Predictive stitching (Part 2: serve a MISS from cached sub-phrases) ---
    # On a full-text MISS, binary-search cached prefix/suffix, synth only the
    # gaps, cross-fade at seams. Skipped below the coverage gate.
    predictive_stitch_enabled: bool = True
    predictive_stitch_min_coverage: float = 0.5  # min cached fraction to stitch

    @property
    def configured_providers(self) -> list[str]:
        """Providers whose required credential is present at startup."""
        live: list[str] = []
        if self.cartesia_api_key:
            live.append("cartesia")
        if self.sarvam_api_key:
            live.append("sarvam")
        if self.elevenlabs_api_key:
            live.append("elevenlabs")
        if self.elevenlabs_indian_residency_api_key:
            live.append("elevenlabs-in")
        if self.google_credentials_json or self.google_credentials_path:
            live.append("gemini")
        return live


settings = Settings()
