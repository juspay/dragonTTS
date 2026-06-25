"""HERMETIC end-to-end test for dragonTTS caching + persistence.

PROVES, with NO real external API, that:
  1. A /tts/bytes MISS synthesizes via the ElevenLabs HTTP endpoint exactly once,
     caches the result, and a repeat identical request is a HIT (no re-synth).
  2. /tts/stream MISS streams audio via the ElevenLabs WebSocket pool, caches it,
     and a repeat is a HIT.
  3. The audio + metadata are PERSISTED to the SQLite DB (cache_entries,
     provider_totals, metrics_daily) and the blob file exists on disk with a size
     matching size_bytes.

How it stays hermetic:
  - A real uvicorn server hosts the real dragonTTS FastAPI app (real lifespan,
    real registry, real cache service, real SQLite store, real blob store) on a
    free localhost port.
  - A mock ElevenLabs server (tests/mock_elevenlabs.py) serves the HTTP synth +
    WS multi-stream-input endpoints on another localhost port.
  - Env is set BEFORE any `app.*` import so ``settings = Settings()`` (run at
    app.core.config import time) picks up the mock host + a dummy key, and only
    elevenlabs is configured (all other provider creds empty).
  - DB_PATH + BLOB_DIR point at a fresh temp dir; the path is printed at the end.

Run:  uv run python tests/e2e_mock_cache.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

# --- 1. Locate repo root so `app` is importable --------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))  # so `import mock_elevenlabs` works


# --- 2. Pick free ports BEFORE setting env -------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


MOCK_PORT = _free_port()
APP_PORT = _free_port()
TMP_DIR = Path(tempfile.mkdtemp(prefix="dragontts_e2e_"))
DB_PATH = TMP_DIR / "e2e.db"
BLOB_DIR = TMP_DIR / "blobs"

# --- 3. Set env BEFORE importing app.* (settings = Settings() runs at import) --
# Only elevenlabs is configured: every other provider key is explicitly empty.
os.environ["ELEVENLABS_INDIAN_RESIDENCY_API_KEY"] = "mock-key"
os.environ["ELEVENLABS_INDIAN_RESIDENCY_BASE_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
os.environ["DB_PATH"] = str(DB_PATH)
os.environ["BLOB_DIR"] = str(BLOB_DIR)
# Empty/absent creds for the other providers so the registry builds elevenlabs only.
os.environ["CARTESIA_API_KEY"] = ""
os.environ["SARVAM_API_KEY"] = ""
os.environ["GOOGLE_CREDENTIALS_JSON"] = ""
os.environ["GOOGLE_CREDENTIALS_PATH"] = ""
# Keep the WS pool small (1 socket) so startup is fast but the stream path still
# exercises the warm multi-context socket (the real differentiator vs HTTP).
os.environ["ELEVENLABS_STREAM_POOL_SIZE"] = "1"
# Predictive warming/stitching would synth extra sub-phrases and muddy the
# "called once per phrase" assertion; disable for a clean MISS/HIT proof.
os.environ["PREDICTIVE_WARM_ENABLED"] = "false"
os.environ["PREDICTIVE_STITCH_ENABLED"] = "false"
os.environ["PREDICTIVE_STITCH_STREAM_ENABLED"] = "false"
# Write-behind metrics defers hit_count/metrics_daily DB writes up to
# metrics_flush_interval_ms (500ms) and only guarantees them on graceful
# shutdown (cache.stop()). This script queries the DB before shutdown to
# prove persistence, so force synchronous metrics writes for determinism.
os.environ["METRICS_WRITE_BEHIND_ENABLED"] = "false"

# NOW it is safe to import app modules and the mock.
from tests import mock_elevenlabs  # noqa: E402

import httpx  # noqa: E402
import uvicorn  # noqa: E402

# Import AFTER env is set so Settings() reads the env above.
from app.main import app  # noqa: E402
from app.core.config import settings  # noqa: E402

VOICE_ID = "fG9s0SXJb213f4UxVHyG"
MODEL_ID = "elevenlabs:eleven_flash_v2_5"


# --- 4. Server boot helpers ----------------------------------------------------
def _boot_uvicorn(port: int) -> uvicorn.Server:
    """Run the real dragonTTS app on localhost:port in a daemon thread."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        loop="asyncio",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]
    started = threading.Event()

    def _spawn(loop_ready: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_ready.set()
        server.run()

    t = threading.Thread(target=_spawn, args=(started,), name="dragontts-uvicorn", daemon=True)
    t.start()
    started.wait(timeout=5.0)
    return server


def _wait_for_health(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                return
            last = f"status={r.status_code} body={r.text[:120]}"
        except Exception as e:
            last = repr(e)
        time.sleep(0.1)
    raise RuntimeError(f"dragonTTS did not become healthy on :{port} ({last})")


# --- 5. Assertion helpers ------------------------------------------------------
def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"    [PASS] {msg}")


def _bytes_request(transcript: str, *, encoding: str = "mulaw", sample_rate: int = 8000) -> dict:
    return {
        "model_id": MODEL_ID,
        "transcript": transcript,
        "voice": {"id": VOICE_ID},
        "language": "en",
        "output_format": {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        },
    }


# --- 6. The test ---------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("HERMETIC e2e: dragonTTS cache + persistence against a MOCK ElevenLabs")
    print("=" * 78)
    print(f"  mock base_url : http://127.0.0.1:{MOCK_PORT}")
    print(f"  app  base_url : http://127.0.0.1:{APP_PORT}")
    print(f"  configured    : {settings.configured_providers}")
    _ok(settings.configured_providers == ["elevenlabs"],
        f"only elevenlabs configured (got {settings.configured_providers})")
    _ok(settings.elevenlabs_indian_residency_base_url == f"http://127.0.0.1:{MOCK_PORT}",
        f"residency base_url points at mock ({settings.elevenlabs_indian_residency_base_url})")
    _ok(settings.db_path == str(DB_PATH), f"DB_PATH -> temp ({settings.db_path})")
    _ok(settings.blob_dir == str(BLOB_DIR), f"BLOB_DIR -> temp ({settings.blob_dir})")

    # Start the mock + the real app.
    mock_elevenlabs.reset_counters()
    mock = mock_elevenlabs.start(MOCK_PORT)
    server = _boot_uvicorn(APP_PORT)
    try:
        _wait_for_health(APP_PORT)
        print(f"  health        : OK ({httpx.get(f'http://127.0.0.1:{APP_PORT}/health').json()})")

        with httpx.Client(base_url=f"http://127.0.0.1:{APP_PORT}", timeout=30.0) as cli:
            # ---------- /tts/bytes : MISS then HIT ----------
            print("\n[1] /tts/bytes — MISS")
            phrase = "hello mock world"
            r1 = cli.post("/tts/bytes", json=_bytes_request(phrase))
            _ok(r1.status_code == 200, f"MISS returns 200 (got {r1.status_code})")
            _ok(len(r1.content) > 0, "MISS body is non-empty")
            _ok(r1.headers.get("X-Cache") == "MISS",
                f"X-Cache == MISS (got {r1.headers.get('X-Cache')!r})")
            bytes_cache_key = r1.headers.get("X-Cache-Key")
            http_after_miss = mock_elevenlabs.http_call_count()
            _ok(http_after_miss == 1,
                f"mock HTTP called exactly once on MISS (got {http_after_miss})")

            print("[2] /tts/bytes — HIT (identical)")
            r2 = cli.post("/tts/bytes", json=_bytes_request(phrase))
            _ok(r2.status_code == 200, f"HIT returns 200 (got {r2.status_code})")
            _ok(r2.headers.get("X-Cache") == "HIT",
                f"X-Cache == HIT (got {r2.headers.get('X-Cache')!r})")
            _ok(r2.content == r1.content, "HIT body byte-identical to MISS body")
            http_after_hit = mock_elevenlabs.http_call_count()
            _ok(http_after_hit == 1,
                f"mock HTTP NOT called again on HIT (still {http_after_hit})")

            # ---------- /tts/stream : MISS then HIT ----------
            # Request NATIVE format (pcm_s16le@16000) so the stream path forwards
            # live WS chunks (the real differentiator) and tees the full clip to
            # the cache. mulaw@8000 would fall back to one-shot synth+convert.
            print("\n[3] /tts/stream — MISS (live WS forward + tee-to-cache)")
            stream_phrase = "streaming hermetic proof phrase"
            stream_req = _bytes_request(stream_phrase, encoding="pcm_s16le", sample_rate=16000)
            with cli.stream("POST", "/tts/stream", json=stream_req) as r3:
                _ok(r3.status_code == 200, f"stream MISS returns 200 (got {r3.status_code})")
                sc = r3.headers.get("X-Cache")
                stream_cache_key = r3.headers.get("X-Cache-Key")
                stream_body = r3.read()
            _ok(sc == "MISS", f"X-Cache == MISS (got {sc!r})")
            _ok(len(stream_body) > 0, "stream body is non-empty")

            # Give the tee-to-cache store + metrics a moment to flush.
            time.sleep(0.3)

            print("[4] /tts/stream — HIT")
            with cli.stream("POST", "/tts/stream", json=stream_req) as r4:
                _ok(r4.status_code == 200, f"stream HIT returns 200 (got {r4.status_code})")
                sc2 = r4.headers.get("X-Cache")
                hit_body = r4.read()
            _ok(sc2 == "HIT", f"X-Cache == HIT (got {sc2!r})")
            _ok(hit_body == stream_body, "stream HIT body byte-identical to MISS body")

            # ---------- DB PERSISTENCE PROOF ----------
            print("\n[5] SQLite + filesystem persistence PROOF")
            # Wait for any deferred writer-thread commit (autocommit, but be safe).
            time.sleep(0.3)
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                ce = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?", (bytes_cache_key,)
                ).fetchone()
                _ok(ce is not None, "cache_entries row exists for the /tts/bytes phrase")
                ce = dict(ce) if ce else {}
                print("    cache_entries (bytes phrase):")
                for k in ("provider", "encoding", "sample_rate", "size_bytes",
                          "storage_path", "hit_count", "voice_id", "model", "language"):
                    print(f"      {k:<14}= {ce.get(k)!r}")
                _ok(ce.get("provider") == "elevenlabs",
                    f"provider == elevenlabs (got {ce.get('provider')!r})")
                _ok(ce.get("encoding") == "pcm_s16le",
                    f"native encoding == pcm_s16le (got {ce.get('encoding')!r})")
                _ok(ce.get("sample_rate") == 16000,
                    f"native sample_rate == 16000 (got {ce.get('sample_rate')!r})")
                _ok(isinstance(ce.get("size_bytes"), int) and ce.get("size_bytes") > 0,
                    f"size_bytes > 0 (got {ce.get('size_bytes')!r})")
                _ok(bool(ce.get("storage_path")),
                    f"storage_path not null (got {ce.get('storage_path')!r})")
                _ok(ce.get("hit_count", 0) >= 1,
                    f"hit_count >= 1 after the HIT (got {ce.get('hit_count')!r})")

                # Blob file on disk matches size_bytes.
                blob_abs = BLOB_DIR / ce["storage_path"]
                on_disk = blob_abs.stat().st_size
                print(f"    blob file       : {blob_abs}")
                print(f"      on-disk bytes = {on_disk}  | size_bytes = {ce['size_bytes']}")
                _ok(blob_abs.exists(), "blob file exists on disk")
                _ok(on_disk == ce["size_bytes"],
                    f"on-disk size == size_bytes ({on_disk} == {ce['size_bytes']})")

                # Stream phrase row too (exercises the WS/tee path's persistence).
                srow = conn.execute(
                    "SELECT size_bytes, storage_path, hit_count FROM cache_entries WHERE key = ?",
                    (stream_cache_key,),
                ).fetchone()
                _ok(srow is not None, "cache_entries row exists for the /tts/stream phrase")
                if srow:
                    sblob = BLOB_DIR / srow["storage_path"]
                    print(f"    stream blob     : {sblob} "
                          f"(size {sblob.stat().st_size if sblob.exists() else 'MISSING'})")
                    _ok(sblob.exists(), "stream blob file exists on disk")
                    _ok(sblob.stat().st_size == srow["size_bytes"],
                        f"stream blob size == size_bytes ({sblob.stat().st_size} == {srow['size_bytes']})")

                # provider_totals for elevenlabs.
                pt = conn.execute(
                    "SELECT * FROM provider_totals WHERE provider = 'elevenlabs'"
                ).fetchone()
                pt = dict(pt) if pt else {}
                print("    provider_totals (elevenlabs):", pt)
                _ok(pt.get("entries", 0) >= 2,
                    f"entries >= 2 (bytes + stream) (got {pt.get('entries')!r})")
                _ok(pt.get("total_bytes", 0) > 0,
                    f"total_bytes > 0 (got {pt.get('total_bytes')!r})")

                # metrics_daily rollup (today's row).
                md = conn.execute(
                    "SELECT * FROM metrics_daily ORDER BY date DESC LIMIT 1"
                ).fetchone()
                md = dict(md) if md else {}
                print("    metrics_daily   :", md)
                _ok(md.get("requests", 0) >= 4,
                    f"requests >= 4 (2 bytes + 2 stream) (got {md.get('requests')!r})")
                _ok(md.get("hits", 0) >= 2,
                    f"hits >= 2 (got {md.get('hits')!r})")
                _ok(md.get("misses", 0) >= 2,
                    f"misses >= 2 (got {md.get('misses')!r})")
                _ok(md.get("synth_calls", 0) >= 2,
                    f"synth_calls >= 2 (got {md.get('synth_calls')!r})")
            finally:
                conn.close()

            print("\n[6] mock counters (PROOF: one synth per unique phrase)")
            print(f"    HTTP call count : {mock_elevenlabs.http_call_count()}  (expect 1)")
            print(f"    WS   call count : {mock_elevenlabs.ws_call_count()}  "
                  f"(>=1 warm connect + 1 stream)")
    finally:
        server.should_exit = True
        mock.stop()
        time.sleep(0.5)

    print("\n" + "=" * 78)
    print(f"ALL GREEN. Temp data dir left for inspection: {TMP_DIR}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print(f"\n*** ASSERTION FAILED ***\n{e}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        import traceback
        print(f"\n*** ERROR ***\n{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
