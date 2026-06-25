"""End-to-end proof that synthesized audio is CACHED and PERSISTED to the
SQLite DB + blob filesystem.

Boots the REAL dragonTTS FastAPI app on a FRESH, KNOWN sqlite DB + blob dir
(env vars are set BEFORE importing app.main so the app's Settings() picks them
up over .env), drives it over a genuine TCP loopback with httpx (NOT TestClient),
and then introspects the sqlite file read-only to prove:

  * cache_entries has a row per unique phrase, with the blob really on disk
    (os.path.getsize(storage_path) == size_bytes)
  * provider_totals is CONSISTENT with cache_entries:
        sum(entries)   == COUNT(*) over cache_entries
        sum(total_bytes) == sum(size_bytes) over cache_entries
  * metrics_daily has requests/hits/misses/synth_calls > 0
  * MISS -> HIT: a second identical request returns X-Cache: HIT, does NOT
    create a duplicate row, and bumps hit_count.

Run:  uv run python tests/verify_real_cache_db.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time

# Ensure the repo root (parent of tests/) is importable, the same way
# pyproject's `pythonpath = ["."]` makes it for pytest.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# 1. Fresh, KNOWN DB + blob dir. Set env vars BEFORE importing app.main so the
#    module-level `settings = Settings()` in app.core.config reads THEM (process
#    env takes precedence over the .env file under pydantic-settings). Delete
#    both first so the run starts from a truly empty store.
# ---------------------------------------------------------------------------
DB_PATH = "/tmp/dragontts_verify.db"
BLOB_DIR = "/tmp/dragontts_verify_blobs"

for p in (DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"):
    if os.path.exists(p):
        os.remove(p)
if os.path.isdir(BLOB_DIR):
    for root, _dirs, files in os.walk(BLOB_DIR, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        # remove empty dirs
    import shutil
    shutil.rmtree(BLOB_DIR)

os.environ["DB_PATH"] = DB_PATH
os.environ["BLOB_DIR"] = BLOB_DIR
# Keep predictive warming/stitching OFF for a deterministic per-phrase cache:
# we want exactly one cache row per unique phrase, not substring sub-rows, so
# the COUNT(*) vs unique-phrase assertion is exact.
os.environ["PREDICTIVE_WARM_ENABLED"] = "false"
os.environ["PREDICTIVE_STITCH_ENABLED"] = "false"
os.environ["PREDICTIVE_STITCH_STREAM_ENABLED"] = "false"
# Write-behind metrics defers hit_count/metrics_daily DB writes up to
# metrics_flush_interval_ms (500ms) and only guarantees them on graceful
# shutdown (cache.stop()). This script queries the DB to prove persistence
# (incl. hit_count bumped by the HIT), so force synchronous metrics writes
# for determinism.
os.environ["METRICS_WRITE_BEHIND_ENABLED"] = "false"

# Now import the app (Settings() loads DB_PATH/BLOB_DIR from our env vars).
import httpx
import uvicorn

from app.core.config import PROVIDER_DEFAULTS, settings  # noqa: E402
from app.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Per-provider test phrases + model/voice from PROVIDER_DEFAULTS.
#    Only test providers that are actually configured (key present at startup).
# ---------------------------------------------------------------------------
TEST_PLAN = {
    "cartesia": {
        "model_id": "cartesia:sonic-3.5",
        "voice": PROVIDER_DEFAULTS["cartesia"]["voice_id"],
        "language": PROVIDER_DEFAULTS["cartesia"]["language"],
    },
    "sarvam": {
        "model_id": "sarvam:bulbul:v3",
        "voice": PROVIDER_DEFAULTS["sarvam"]["voice_id"],
        "language": PROVIDER_DEFAULTS["sarvam"]["language"],
    },
    "elevenlabs": {
        "model_id": "elevenlabs:eleven_flash_v2_5",
        "voice": PROVIDER_DEFAULTS["elevenlabs"]["voice_id"],
        "language": PROVIDER_DEFAULTS["elevenlabs"]["language"],
    },
}

# Unique, run-stamped phrases so re-runs never collide with a leftover cache
# (the store is wiped above, but the stamp also disambiguates within a run).
STAMP = str(int(time.time() * 1000))[-8:]
PHRASES = {
    "cartesia": f"Cartesia cache persistence test number {STAMP} alpha.",
    "sarvam": f"Sarvam cache persistence test number {STAMP} bravo.",
    "elevenlabs": f"ElevenLabs cache persistence test number {STAMP} charlie.",
}
STREAM_PHRASES = {
    "cartesia": f"Streaming a fresh phrase for cartesia number {STAMP} delta.",
    "sarvam": f"Streaming a fresh phrase for sarvam number {STAMP} echo.",
    "elevenlabs": f"Streaming a fresh phrase for elevenlabs number {STAMP} foxtrot.",
}

TELEPHONY_OF = {"container": "raw", "encoding": "mulaw", "sample_rate": 8000}

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}")
    if not cond:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# 3. Boot the real server in a daemon thread on a free port.
# ---------------------------------------------------------------------------
def find_free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


PORT = find_free_port()
BASE = f"http://127.0.0.1:{PORT}"

config = uvicorn.Config(
    "app.main:app", host="127.0.0.1", port=PORT, log_level="warning"
)
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()

print(f"Booting real dragonTTS server on {BASE} ...")
print(f"  DB_PATH  = {DB_PATH}")
print(f"  BLOB_DIR = {BLOB_DIR}")
print(f"  configured_providers = {settings.configured_providers}")
print("-" * 78)

# Poll /health until 200 (timeout ~20s).
deadline = time.monotonic() + 20.0
healthy = False
last_err = None
while time.monotonic() < deadline:
    if not server.started:
        time.sleep(0.1)
        continue
    try:
        with httpx.Client(base_url=BASE, timeout=5.0) as c:
            r = c.get("/health")
            if r.status_code == 200:
                healthy = True
                print(f"  /health -> 200: {r.json()}")
                break
    except Exception as e:  # not up yet
        last_err = e
        time.sleep(0.2)

if not healthy:
    print(f"FATAL: server did not become healthy within 20s (last err: {last_err})")
    server.should_exit = True
    sys.exit(1)

print("-" * 78)

# Providers actually live in this process (key present).
configured = settings.configured_providers
providers_to_test = [p for p in ("cartesia", "sarvam", "elevenlabs") if p in configured]

# ---------------------------------------------------------------------------
# 4. Drive the endpoints with httpx (real network round-trip).
#    For each provider: /tts/bytes MISS, /tts/bytes HIT (identical), one
#    /tts/stream MISS for a fresh phrase. Telephony output_format.
# ---------------------------------------------------------------------------
print("[1/3] Driving /tts/bytes (MISS then HIT) + /tts/stream per provider")
print("-" * 78)

per_provider_rows: dict[str, dict] = {}

with httpx.Client(base_url=BASE, timeout=60.0) as client:
    for prov in providers_to_test:
        plan = TEST_PLAN[prov]
        phrase = PHRASES[prov]
        body = {
            "model_id": plan["model_id"],
            "transcript": phrase,
            "voice": {"id": plan["voice"]},
            "language": plan["language"],
            "output_format": TELEPHONY_OF,
        }
        print(f"  {prov}:")
        info: dict = {"bytes_ok": False, "miss_status": None, "hit_status": None,
                      "miss_bytes": 0, "hit_bytes": 0, "stream_ok": False,
                      "stream_bytes": 0, "err": None}

        # --- MISS ---
        try:
            r1 = client.post("/tts/bytes", json=body)
            info["miss_status"] = r1.headers.get("X-Cache")
            info["miss_bytes"] = len(r1.content)
            ok = (r1.status_code == 200 and len(r1.content) > 0
                  and r1.headers.get("X-Cache") == "MISS")
            info["bytes_ok"] = ok
            check(r1.status_code == 200, f"{prov} MISS status==200 (got {r1.status_code})")
            check(len(r1.content) > 0, f"{prov} MISS non-empty audio ({len(r1.content)} bytes)")
            check(r1.headers.get("X-Cache") == "MISS",
                  f"{prov} MISS X-Cache==MISS (got {r1.headers.get('X-Cache')!r})")
        except Exception as e:
            info["err"] = f"MISS: {type(e).__name__}: {e}"
            print(f"    ERROR on MISS: {info['err']}")

        # --- HIT (identical) ---
        try:
            r2 = client.post("/tts/bytes", json=body)
            info["hit_status"] = r2.headers.get("X-Cache")
            info["hit_bytes"] = len(r2.content)
            check(r2.status_code == 200, f"{prov} HIT status==200 (got {r2.status_code})")
            check(r2.headers.get("X-Cache") == "HIT",
                  f"{prov} HIT X-Cache==HIT (got {r2.headers.get('X-Cache')!r})")
            if r1.status_code == 200 and r2.status_code == 200:
                check(r1.content == r2.content,
                      f"{prov} HIT bytes identical to MISS ({len(r1.content)} == {len(r2.content)})")
        except Exception as e:
            info["err"] = f"HIT: {type(e).__name__}: {e}"
            print(f"    ERROR on HIT: {info['err']}")

        # --- STREAM (fresh phrase) ---
        sbody = dict(body)
        sbody["transcript"] = STREAM_PHRASES[prov]
        try:
            with client.stream("POST", "/tts/stream", json=sbody) as r3:
                check(r3.status_code == 200,
                      f"{prov} STREAM status==200 (got {r3.status_code})")
                chunks = b"".join(r3.iter_bytes())
                info["stream_bytes"] = len(chunks)
                info["stream_ok"] = r3.status_code == 200 and len(chunks) > 0
                check(len(chunks) > 0,
                      f"{prov} STREAM non-empty bytes ({len(chunks)} bytes)")
                check(r3.headers.get("X-Cache") in ("MISS", "MISS-STITCH", "HIT"),
                      f"{prov} STREAM X-Cache in (MISS,MISS-STITCH,HIT) "
                      f"(got {r3.headers.get('X-Cache')!r})")
        except Exception as e:
            info["err"] = f"STREAM: {type(e).__name__}: {e}"
            print(f"    ERROR on STREAM: {info['err']}")

        per_provider_rows[prov] = info
        if info["err"]:
            print(f"    !! {prov} had an error: {info['err']}")

print("-" * 78)

# Give any in-flight writes a moment to flush (autocommit, but be safe) before
# we open the DB read-only from THIS thread (a different connection than the
# server's per-thread ones — fine under WAL).
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 5. DB + BLOB introspection (read-only, fresh sqlite3 connection).
# ---------------------------------------------------------------------------
print("[2/3] DB + blob introspection (read-only)")
print("-" * 78)

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

# --- cache_entries: count + total bytes ---
row = conn.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM cache_entries").fetchone()
ce_count, ce_total = row[0], row[1]
print(f"  cache_entries: count={ce_count}, sum(size_bytes)={ce_total}")

# Number of unique phrases we stored via /tts/bytes MISS (one per working prov)
# PLUS one stream phrase per working prov whose synth succeeded. Counting only
# providers whose MISS actually succeeded keeps the lower bound honest.
working_bytes = [p for p in providers_to_test if per_provider_rows[p]["bytes_ok"]]
working_stream = [p for p in providers_to_test if per_provider_rows[p]["stream_ok"]]
min_expected_rows = len(working_bytes) + len(working_stream)
print(f"  expected >= {min_expected_rows} rows "
      f"({len(working_bytes)} bytes-MISS + {len(working_stream)} stream-MISS)")
check(ce_count >= min_expected_rows,
      f"cache_entries COUNT(*) >= unique phrases ({ce_count} >= {min_expected_rows})")
check(ce_total > 0, f"cache_entries sum(size_bytes) > 0 ({ce_total})")

# --- provider_totals: consistency vs cache_entries ---
pt_rows = conn.execute(
    "SELECT provider, entries, total_bytes FROM provider_totals ORDER BY provider"
).fetchall()
print("\n  provider_totals:")
for r in pt_rows:
    print(f"    provider={r['provider']:12s} entries={r['entries']} "
          f"total_bytes={r['total_bytes']}")
pt_sum_entries = sum(r["entries"] for r in pt_rows)
pt_sum_bytes = sum(r["total_bytes"] for r in pt_rows)
print(f"  SUM over provider_totals: entries={pt_sum_entries}, total_bytes={pt_sum_bytes}")
print(f"  cache_entries           : count={ce_count}, sum(size_bytes)={ce_total}")
check(pt_sum_entries == ce_count,
      f"sum(provider_totals.entries) == cache_entries COUNT(*) "
      f"({pt_sum_entries} == {ce_count})")
check(pt_sum_bytes == ce_total,
      f"sum(provider_totals.total_bytes) == cache_entries sum(size_bytes) "
      f"({pt_sum_bytes} == {ce_total})")

# Each used provider has >=1 entry.
for prov in working_bytes:
    match = [r for r in pt_rows if r["provider"] == prov]
    check(bool(match) and match[0]["entries"] >= 1,
          f"provider_totals has {prov} with entries>=1 "
          f"(entries={match[0]['entries'] if match else 'MISSING'})")

# --- metrics_daily ---
print("\n  metrics_daily:")
md_rows = conn.execute(
    "SELECT date, requests, hits, misses, synth_calls, bytes_served "
    "FROM metrics_daily ORDER BY date"
).fetchall()
for r in md_rows:
    print(f"    date={r['date']} requests={r['requests']} hits={r['hits']} "
          f"misses={r['misses']} synth_calls={r['synth_calls']} "
          f"bytes_served={r['bytes_served']}")
tot_req = sum(r["requests"] for r in md_rows)
tot_hits = sum(r["hits"] for r in md_rows)
tot_miss = sum(r["misses"] for r in md_rows)
tot_synth = sum(r["synth_calls"] for r in md_rows)
check(tot_req > 0, f"metrics_daily SUM(requests) > 0 ({tot_req})")
check(tot_hits > 0, f"metrics_daily SUM(hits) > 0 ({tot_hits})")
check(tot_miss > 0, f"metrics_daily SUM(misses) > 0 ({tot_miss})")
check(tot_synth > 0, f"metrics_daily SUM(synth_calls) > 0 ({tot_synth})")

# --- 2 sample rows: detail + blob-on-disk size match ---
print("\n  sample cache rows (blob on-disk verification):")
sample = conn.execute(
    "SELECT key, provider, voice_id, model, encoding, sample_rate, "
    "size_bytes, hit_count, storage_path FROM cache_entries LIMIT 2"
).fetchall()
for r in sample:
    k = r["key"]
    # storage_path is stored RELATIVE to BLOB_DIR (e.g. "2b/ab/<key>"); resolve
    # it against BLOB_DIR to get the absolute on-disk location.
    abs_path = os.path.join(BLOB_DIR, r["storage_path"])
    disk_exists = os.path.exists(abs_path)
    disk_size = os.path.getsize(abs_path) if disk_exists else -1
    print(f"    key={k[:16]}… provider={r['provider']:12s} voice_id={r['voice_id'][:12]}… "
          f"model={r['model']}")
    print(f"      encoding={r['encoding']} sample_rate={r['sample_rate']} "
          f"size_bytes={r['size_bytes']} hit_count={r['hit_count']}")
    print(f"      storage_path={r['storage_path']} (rel to BLOB_DIR)")
    print(f"      abs blob path: {abs_path}")
    print(f"      blob on disk: exists={disk_exists} getsize={disk_size} "
          f"(db size_bytes={r['size_bytes']}) "
          f"{'MATCH' if disk_exists and disk_size == r['size_bytes'] else 'MISMATCH'}")
    check(disk_exists, f"blob exists on disk ({r['storage_path']})")
    check(disk_exists and disk_size == r["size_bytes"],
          f"blob getsize == size_bytes ({disk_size} == {r['size_bytes']})")

# --- HIT did NOT create a duplicate row, and bumped hit_count ---
# The MISS-then-HIT phrase's key should appear exactly once with hit_count>=1.
print("\n  MISS->HIT: no duplicate row + hit_count bumped:")
for prov in working_bytes:
    phrase = PHRASES[prov]
    plan = TEST_PLAN[prov]
    # Re-derive the key the same way the app does to find the row.
    from app.cache.key import canonical_params, hash_key
    provider_name, model = plan["model_id"].split(":", 1)
    params_canon = canonical_params(provider_name, {})
    key = hash_key(
        text=phrase, provider=provider_name, voice_id=plan["voice"],
        model=model, language=plan["language"], params_canonical=params_canon,
    )
    rows = conn.execute(
        "SELECT hit_count FROM cache_entries WHERE key = ?", (key,)
    ).fetchall()
    n = len(rows)
    hc = rows[0]["hit_count"] if rows else None
    print(f"    {prov:12s}: rows_for_key={n} hit_count={hc} (>=1 expected)")
    check(n == 1, f"{prov} HIT created no duplicate row (rows_for_key == 1, got {n})")
    check(hc is not None and hc >= 1, f"{prov} HIT bumped hit_count (>=1, got {hc})")

conn.close()

# ---------------------------------------------------------------------------
# 6. Shutdown + report.
# ---------------------------------------------------------------------------
print("-" * 78)
print("[3/3] Shutdown")
server.should_exit = True
thread.join(timeout=10.0)
print(f"  server stopped (thread alive={thread.is_alive()})")
print("=" * 78)

print("SUMMARY")
print(f"  configured_providers: {configured}")
print(f"  tested providers:     {providers_to_test}")
for prov in providers_to_test:
    info = per_provider_rows[prov]
    if info["err"]:
        print(f"  {prov:12s}: ERROR -> {info['err']}")
    else:
        print(f"  {prov:12s}: MISS X-Cache={info['miss_status']} "
              f"({info['miss_bytes']}B) | HIT X-Cache={info['hit_status']} "
              f"({info['hit_bytes']}B) | STREAM ok={info['stream_ok']} "
              f"({info['stream_bytes']}B)")
print(f"  cache_entries: count={ce_count} sum(size_bytes)={ce_total}")
print(f"  provider_totals: sum(entries)={pt_sum_entries} sum(total_bytes)={pt_sum_bytes} "
      f"(== COUNT(*) / sum(size_bytes): "
      f"{pt_sum_entries == ce_count and pt_sum_bytes == ce_total})")
print(f"  metrics_daily: requests={tot_req} hits={tot_hits} misses={tot_miss} "
      f"synth_calls={tot_synth}")
print("=" * 78)

if FAILURES:
    print(f"RESULT: FAIL ({len(FAILURES)} check(s) failed)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
