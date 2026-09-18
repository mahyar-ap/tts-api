#!/usr/bin/env bash
# Install persian-tts-api on a local Ubuntu GPU host (e.g. 1080 Ti).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
MANATTS_TARGET="${MANATTS_TARGET:-.manatts-packages}"
# Pascal (1080 Ti) often works better with cu118; override if your driver supports cu124.
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.11 first." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. Install NVIDIA drivers before CUDA Torch." >&2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url "$PYTORCH_INDEX_URL"
python -m pip install \
  -r requirements.txt \
  -c constraints-tested.txt
python -m pip install \
  -r requirements-chatterbox-deps.txt \
  -c constraints-tested.txt
python -m pip install \
  --no-deps \
  -r requirements-chatterbox-package.txt
python scripts/check_dependencies.py

case "$MANATTS_TARGET" in
  "" | "/" | "." | "..")
    echo "MANATTS_TARGET must name a dedicated package directory." >&2
    exit 1
    ;;
esac
mkdir -p "$MANATTS_TARGET"
find "$MANATTS_TARGET" -mindepth 1 -delete
python -m pip install \
  --target "$MANATTS_TARGET" \
  --ignore-installed \
  --no-build-isolation \
  -r requirements-manatts.txt
PYTHONPATH="$(realpath "$MANATTS_TARGET")" \
  python -S scripts/check_manatts_dependencies.py "$MANATTS_TARGET"

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device0={torch.cuda.get_device_name(0)}")
PY

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — set HF_TOKEN before loading Chatterbox."
fi

echo "Server dependency installation completed successfully."
echo "Next: edit .env (HF_TOKEN, DEVICE=cuda, MODEL_UNLOAD_POLICY=single-model),"
echo "optional ManaTTS assets via scripts/setup_manatts.py, then scripts/run_api.sh"
