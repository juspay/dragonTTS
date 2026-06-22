#!/usr/bin/env bash
# Local smoke test for DragonTTS.
# Assumes the server is running:  uv run uvicorn app.main:app --port 8000
# and a provider key is set in .env (e.g. CARTESIA_API_KEY=...).
#
# Override knobs via env, e.g.:
#   PROVIDER=sarvam MODEL=bulbul:v3 VOICE=shreya ENC=mulaw RATE=8000 bash scripts/local_smoke.sh
set -euo pipefail

HOST="${DRAGONTTS_HOST:-http://localhost:8000}"
PROVIDER="${PROVIDER:-cartesia}"
MODEL="${MODEL:-sonic-3.5}"
VOICE="${VOICE:-bec003e2-3cb3-429c-8468-206a393c67ad}"
ENC="${ENC:-pcm_s16le}"
RATE="${RATE:-16000}"
TEXT="${TEXT:-thank you for your order}"

BODY="{\"model_id\":\"$PROVIDER:$MODEL\",\"transcript\":\"$TEXT\",\"voice\":{\"id\":\"$VOICE\"},\"language\":\"en\",\"output_format\":{\"container\":\"raw\",\"encoding\":\"$ENC\",\"sample_rate\":$RATE}}"

echo "== GET /health =="
curl -fsS "$HOST/health"; echo
echo "== GET /providers =="
curl -fsS "$HOST/providers"; echo

echo "== POST /tts/check (before — expect cached:false) =="
curl -fsS -X POST "$HOST/tts/check" -H "Content-Type: application/json" -d "$BODY"; echo

echo "== POST /tts/create (synth via provider + store) =="
curl -fsS -X POST "$HOST/tts/create" -H "Content-Type: application/json" -d "$BODY"; echo

echo "== POST /tts/check (after — expect cached:true) =="
curl -fsS -X POST "$HOST/tts/check" -H "Content-Type: application/json" -d "$BODY"; echo

echo "== POST /tts/bytes #1 (expect HIT) =="
curl -fsS -X POST "$HOST/tts/bytes" -H "Content-Type: application/json" -d "$BODY" \
  -o /tmp/dtts1.bin -D - | grep -i '^x-cache'
echo "== POST /tts/bytes #2 (expect HIT) =="
curl -fsS -X POST "$HOST/tts/bytes" -H "Content-Type: application/json" -d "$BODY" \
  -o /tmp/dtts2.bin -D - | grep -i '^x-cache'

echo "== GET /stats =="
curl -fsS "$HOST/stats"; echo
echo "== GET /cache (list) =="
curl -fsS "$HOST/cache"; echo

echo
echo "audio bytes: $(wc -c < /tmp/dtts1.bin)  (identical across calls: $([ "$(cat /tmp/dtts1.bin)" = "$(cat /tmp/dtts2.bin)" ] && echo yes || echo NO))"
