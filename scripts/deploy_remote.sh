#!/usr/bin/env bash
# Rsync this repo to mahyar@10.1.20.25 (or REMOTE override).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-mahyar@10.1.20.25}"
REMOTE_DIR="${REMOTE_DIR:-~/persian-tts-api}"

rsync -avz --delete \
  --exclude '.venv/' \
  --exclude '.manatts-packages/' \
  --exclude 'model_cache/' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.git/' \
  --exclude 'third_party/' \
  "$ROOT/" \
  "${REMOTE}:${REMOTE_DIR}/"

echo "Synced to ${REMOTE}:${REMOTE_DIR}"
echo "On the server: cd ${REMOTE_DIR} && bash scripts/install_server.sh && bash scripts/run_api.sh"
