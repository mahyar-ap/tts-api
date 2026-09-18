#!/usr/bin/env bash
set -euo pipefail

# Lightning supports one persistent Python environment per Studio. ManaTTS is
# therefore installed into a private package directory for its -S subprocess.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
MANATTS_TARGET="${MANATTS_TARGET:-.manatts-packages}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url "$PYTORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install \
  -r requirements.txt \
  -c constraints-tested.txt
"$PYTHON_BIN" -m pip install \
  -r requirements-chatterbox-deps.txt \
  -c constraints-tested.txt
"$PYTHON_BIN" -m pip install \
  --no-deps \
  -r requirements-chatterbox-package.txt
"$PYTHON_BIN" scripts/check_dependencies.py

case "$MANATTS_TARGET" in
  "" | "/" | "." | "..")
    echo "MANATTS_TARGET must name a dedicated package directory." >&2
    exit 1
    ;;
esac
if [[ -e "$MANATTS_TARGET" && ! -d "$MANATTS_TARGET" ]]; then
  echo "MANATTS_TARGET exists but is not a directory." >&2
  exit 1
fi
mkdir -p "$MANATTS_TARGET"
find "$MANATTS_TARGET" -mindepth 1 -delete
"$PYTHON_BIN" -m pip install \
  --target "$MANATTS_TARGET" \
  --ignore-installed \
  --no-build-isolation \
  -r requirements-manatts.txt
PYTHONPATH="$(realpath "$MANATTS_TARGET")" \
  "$PYTHON_BIN" -S scripts/check_manatts_dependencies.py "$MANATTS_TARGET"

echo "Lightning dependency installation completed successfully."
