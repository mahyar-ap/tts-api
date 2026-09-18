#!/usr/bin/env bash
# Quick smoke against a running API (default http://127.0.0.1:8000).
set -euo pipefail

BASE="${TTS_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${1:-chatterbox}"

echo "Health:"
curl -fsS "$BASE/health" | python -m json.tool

echo "Load $MODEL:"
curl -fsS -X POST "$BASE/v1/models/${MODEL}/load" | python -m json.tool

echo "Stream sample (first events):"
curl -NsS -X POST "$BASE/v1/tts/${MODEL}/stream" \
  -H 'Content-Type: application/json' \
  -d '{"text":"سلام. این یک آزمایش است."}' | head -n 20
