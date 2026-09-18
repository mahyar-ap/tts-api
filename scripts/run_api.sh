#!/usr/bin/env bash
# Start the Persian TTS API (single Uvicorn worker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export DEVICE="${DEVICE:-cuda}"
export WORKER_COUNT="${WORKER_COUNT:-1}"
export MODEL_UNLOAD_POLICY="${MODEL_UNLOAD_POLICY:-single-model}"
export API_KEY_ENABLED="${API_KEY_ENABLED:-false}"

exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1
