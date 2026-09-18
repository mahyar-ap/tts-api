# Persian Open-Weight TTS API

A FastAPI service exposing six Persian text-to-speech engines through a bounded
asynchronous job queue. Models load lazily, every engine has a concurrency lock,
and generated PCM-16 WAV files plus job metadata are stored under `OUTPUT_DIR`.

Swagger UI is available at `/docs` and ReDoc at `/redoc`.

## Deployment audit summary

The main environment deliberately uses this tested Persian Chatterbox stack:

- Python 3.11
- `chatterbox-tts==0.1.7`
- `torch==2.6.0`
- `torchaudio==2.6.0`
- `torchvision==0.21.0`
- `transformers==4.52.0`

Chatterbox 0.1.7 package metadata declares `transformers==5.2.0`, which conflicts
with the tested Persian version. The installer therefore installs Chatterbox
with `--no-deps` and then installs its dependencies explicitly. Do not replace
this process with a plain `pip install chatterbox-tts`, because pip may replace
the tested Torch and Transformers packages.

Chatterbox runs in one persistent supervised subprocess. That process alone
imports Chatterbox and initializes its CUDA context. A generation error, worker
timeout, crash, or malformed protocol response causes the API to terminate that
child; the failed job remains isolated, and the next Chatterbox job starts a
fresh worker without restarting FastAPI or disturbing another engine.
Parent and child exchange request-ID-bearing JSON messages on standard
input/output; model diagnostics and tracebacks are reserved for standard error.

ManaTTS uses older ParallelWaveGAN and SciPy integrations. Those packages are
kept in `.manatts-packages` and are not added to the Chatterbox/FastAPI import
path. The API starts ManaTTS with `python3.11 -S` and a private `PYTHONPATH`, so
the model remains loaded in one persistent CPU subprocess while dependency
conflicts stay isolated. This package-layer design follows Lightning's
[one-environment-per-Studio guidance](https://lightning.ai/docs/overview/ai-studio/environment-persistence)
without forcing the legacy libraries into the main runtime.

## Project layout

```text
.
├── app
│   ├── engines
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chatterbox.py
│   │   ├── chatterbox_backend.py
│   │   ├── chatterbox_worker.py
│   │   ├── khadijah_matcha.py
│   │   ├── mana_piper.py
│   │   ├── manatts_backend.py
│   │   ├── manatts_tacotron2.py
│   │   ├── manatts_worker.py
│   │   ├── piper_base.py
│   │   ├── piper_ganji.py
│   │   └── piper_ganji_adabi.py
│   ├── __init__.py
│   ├── config.py
│   ├── jobs.py
│   ├── main.py
│   ├── queue.py
│   ├── registry.py
│   └── schemas.py
├── scripts
│   ├── check_dependencies.py
│   ├── check_manatts_dependencies.py
│   ├── install_lightning.sh
│   ├── setup_manatts.py
│   ├── smoke_chatterbox.py
│   └── smoke_import_engines.py
├── tests
│   ├── test_api.py
│   ├── test_chatterbox.py
│   ├── test_engines.py
│   ├── test_jobs.py
│   └── test_smoke_scripts.py
├── .env.example
├── .gitignore
├── constraints-tested.txt
├── pyproject.toml
├── requirements-chatterbox-deps.txt
├── requirements-chatterbox-package.txt
├── requirements-dev.txt
├── requirements-manatts.txt
└── requirements.txt
```

## Exact Lightning AI installation

Use a fresh Python 3.11 Lightning Studio with a Tesla T4 and an NVIDIA driver
that supports the official PyTorch CUDA 12.4 wheels. Put the project inside the
Studio's persistent `/teamspace/studios/this_studio` directory, then run:

```bash
export TTS_PROJECT_DIR=/teamspace/studios/this_studio/persian-open-weight-tts-api
cd "$TTS_PROJECT_DIR"
export PYTHON_BIN=python3.11
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
export MANATTS_TARGET=.manatts-packages
bash scripts/install_lightning.sh
```

The installer creates:

- The active Studio Python environment: FastAPI, Chatterbox, Piper, and
  sherpa-onnx with the tested CUDA Torch stack.
- `.manatts-packages`: a private CPU Torch, ManaTTS, and ParallelWaveGAN package
  layer used only by the isolated subprocess.

The official matching PyTorch command used for the main environment is:

```bash
python3.11 -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Do not run `pip install -U torch`, `pip install -U transformers`, or a direct
dependency-resolving Chatterbox install afterward.

### Prepare ManaTTS model assets

This command clones the upstream repository, downloads its synthesizer,
reference WAV, and VCTK HiFiGAN vocoder, and applies the required SciPy and
librosa compatibility patches:

```bash
export HF_TOKEN='hf_replace_with_your_read_token'
PYTHONPATH="$TTS_PROJECT_DIR/.manatts-packages" \
  python3.11 -S scripts/setup_manatts.py
```

It creates:

```text
third_party/Persian-MultiSpeaker-Tacotron2/saved_models/final_models/
├── config.yml
├── encoder.pt
├── sample.wav
├── synthesizer.pt
└── vocoder_HiFiGAN.pkl
```

## Exact environment setup

Accept the gated Persian Chatterbox repository conditions before using its Hugging
Face token. For local development with API-key authentication disabled:

```bash
cp .env.example .env
export HF_TOKEN='hf_replace_with_your_read_token'
export DEVICE=cuda
export MAX_TEXT_LENGTH=10000
export MAX_QUEUE_SIZE=100
export WORKER_COUNT=1
export OUTPUT_DIR=outputs
export MODEL_CACHE_DIR=model_cache
export MODEL_UNLOAD_POLICY=single-heavy
export CHATTERBOX_CHUNK_SIZE=300
export CHATTERBOX_STARTUP_TIMEOUT_SECONDS=300
export CHATTERBOX_INFERENCE_TIMEOUT_SECONDS=1800
export JOB_TTL_SECONDS=86400
export JOB_CLEANUP_INTERVAL_SECONDS=300
export API_KEY_ENABLED=false
export MANATTS_REPO_DIR=third_party/Persian-MultiSpeaker-Tacotron2
export MANATTS_PYTHON=python3.11
export MANATTS_PACKAGE_DIR=.manatts-packages
```

To require an API key for every `/v1/` route:

```bash
export API_KEY_ENABLED=true
export API_KEY='replace-with-a-long-random-secret'
export API_KEY_HEADER=X-API-Key
```

`/health`, `/docs`, `/redoc`, and `/openapi.json` remain available without the
key. Supply the configured header for model, submission, status, and audio calls.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | empty | Hugging Face token required by gated Chatterbox weights |
| `MAX_TEXT_LENGTH` | `10000` | Pydantic input limit, capped at 10,000 |
| `MAX_QUEUE_SIZE` | `100` | Maximum number of waiting jobs |
| `WORKER_COUNT` | `1` | Background consumers; keep at 1 for the T4 |
| `OUTPUT_DIR` | `outputs` | Persistent WAV and metadata directory |
| `DEVICE` | `cuda` | Chatterbox device |
| `JOB_TTL_SECONDS` | `86400` | Retention for completed and failed jobs; 0 disables cleanup |
| `JOB_CLEANUP_INTERVAL_SECONDS` | `300` | Cleanup scan interval |
| `API_KEY_ENABLED` | `false` | Protect `/v1/` routes when true |
| `API_KEY` | empty | Expected API key value |
| `API_KEY_HEADER` | `X-API-Key` | Configurable request header |
| `MODEL_CACHE_DIR` | `model_cache` | Persistent Hugging Face cache |
| `MODEL_UNLOAD_POLICY` | `single-heavy` | `none`, `single-heavy`, or `single-model` |
| `RETRY_AFTER_SECONDS` | `5` | Queue-full `Retry-After` value |
| `CHATTERBOX_CHUNK_SIZE` | `300` | Maximum characters per Chatterbox `generate()` call; capped at 300 |
| `CHATTERBOX_STARTUP_TIMEOUT_SECONDS` | `300` | Isolated Chatterbox model startup timeout |
| `CHATTERBOX_INFERENCE_TIMEOUT_SECONDS` | `1800` | Total timeout for every chunk in one Chatterbox job |
| `MANATTS_REPO_DIR` | `third_party/Persian-MultiSpeaker-Tacotron2` | ManaTTS checkout |
| `MANATTS_PYTHON` | `python3.11` | Interpreter used for the isolated worker |
| `MANATTS_PACKAGE_DIR` | `.manatts-packages` | Private ManaTTS package layer |
| `MANATTS_STARTUP_TIMEOUT_SECONDS` | `300` | ManaTTS worker startup timeout |
| `MANATTS_INFERENCE_TIMEOUT_SECONDS` | `300` | ManaTTS request timeout |
| `ESPEAK_NG_DATA_DIR` | automatic | Optional local espeak-ng data directory |
| `SHERPA_NUM_THREADS` | `2` | sherpa-onnx CPU threads |

Keep `WORKER_COUNT=1` on a Tesla T4. Async request handling keeps HTTP responsive,
but it does not make GPU inference safe to run concurrently. Model loading,
inference, and unloading run through `asyncio.to_thread`; every engine also has a
lock. `WORKER_COUNT` controls the API job consumer and is separate from the one
persistent Chatterbox subprocess.

## API workflow

Submission endpoints:

- `POST /v1/tts/chatterbox`
- `POST /v1/tts/manatts-tacotron2`
- `POST /v1/tts/mana-piper`
- `POST /v1/tts/piper-ganji`
- `POST /v1/tts/piper-ganji-adabi`
- `POST /v1/tts/khadijah-matcha`

Example with authentication enabled:

```bash
curl -i -X POST http://127.0.0.1:8000/v1/tts/chatterbox \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -d '{"text":"سلام، این یک آزمایش است."}'
```

The submission returns `202` with `job_id`, `status_url`, and `audio_url`.
Invalid text returns `422`. A full queue returns `429` with `Retry-After`. The
audio route returns `409` before completion, a sanitized `500` after failure,
and `audio/wav` after success. Full third-party exceptions and tracebacks are
logged server-side and are never stored in public job metadata.

Chatterbox accepts the same 10,000-character API maximum as the other models,
but never sends the full request to one upstream generation call. It groups text
at Persian or Latin sentence boundaries, falls back to clause/word boundaries
for oversized sentences, synthesizes chunks sequentially, and joins compatible
PCM WAV chunks with 120 ms of silence. The 300-character default follows the
upstream multilingual demo's input ceiling and can be lowered with
`CHATTERBOX_CHUNK_SIZE`.

Interrupted queued or running jobs are marked failed during startup. The cleanup
task removes expired terminal metadata and WAV files, preventing unbounded output
growth.

## Unit tests and dependency checks

Unit tests use mocks and never download model weights:

```bash
python3.11 -m pip install -r requirements-dev.txt
python3.11 -m pytest -q
```

Check the real installed version set. The custom checker permits only the known
Chatterbox `transformers==5.2.0` metadata mismatch:

```bash
python3.11 scripts/check_dependencies.py
PYTHONPATH="$TTS_PROJECT_DIR/.manatts-packages" \
  python3.11 -S scripts/check_manatts_dependencies.py .manatts-packages
```

Import every engine without loading weights:

```bash
python3.11 scripts/smoke_import_engines.py
```

## Real Chatterbox smoke test

This command downloads and loads the gated model, applies `t3_fa.safetensors` to
`model.t3` with strict loading, and generates a short PCM-16 WAV:

```bash
export HF_TOKEN='hf_replace_with_your_read_token'
export DEVICE=cuda
python3.11 scripts/smoke_chatterbox.py \
  --output outputs/chatterbox-smoke.wav
```

## Start FastAPI

Use one Uvicorn process so the service has one GPU queue and one model registry:

```bash
python3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

For a bare-metal GPU host such as `10.1.20.25`, see [SERVER.md](SERVER.md)
(`scripts/install_server.sh`, `scripts/deploy_remote.sh`, `scripts/run_api.sh`).
