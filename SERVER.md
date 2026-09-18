# Deploy on mahyar@10.1.20.25 (1080 Ti)

## One-time sync + install

### From Windows PowerShell (no WSL bash required)

```powershell
cd C:\Users\mahya\Documents\workspace\persian-tts-api
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_remote.ps1
```

### From Linux / WSL (with bash + rsync)

```bash
cd /path/to/persian-tts-api
bash scripts/deploy_remote.sh
```

Then on the GPU host:

```bash
ssh mahyar@10.1.20.25
cd ~/persian-tts-api
bash scripts/install_server.sh
```

Edit `~/persian-tts-api/.env`:

- `HF_TOKEN=...` (required for default Chatterbox)
- `DEVICE=cuda`
- `MODEL_UNLOAD_POLICY=single-model`
- `WORKER_COUNT=1`
- `API_KEY_ENABLED=false`

Optional systemd:

```bash
sudo cp scripts/persian-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now persian-tts
```

Or foreground:

```bash
bash scripts/run_api.sh
```

Smoke:

```bash
bash scripts/smoke_stream.sh chatterbox
```

## API notes

- Jobs (unchanged): `POST /v1/tts/{model}` → poll → `/v1/jobs/{id}/audio`
- Stream: `POST /v1/tts/{model}/stream` → SSE `chunk` / `done` / `error`
- Load one model: `POST /v1/models/{model}/load` (unloads others under `single-model`)
- Unload: `POST /v1/models/{model}/unload`

Point BankAssist at `TTS_BASE_URL=http://10.1.20.25:8000` with `TTS_ENGINE=chatterbox`.
