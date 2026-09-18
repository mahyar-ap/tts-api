"""Line-oriented worker that owns the Chatterbox CUDA context."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from app.engines.chatterbox_backend import ChatterboxBackend

_PROTOCOL_VERSION = 1


def send(payload: dict[str, Any]) -> None:
    """Write exactly one structured protocol message to standard output."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def log_failure(message: str) -> None:
    """Keep diagnostics and third-party output off the protocol stream."""

    print(message, file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def main() -> int:
    """Load Chatterbox once and process requests until failure or shutdown."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-chunk-length", type=int, required=True)
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is missing in the Chatterbox worker.", file=sys.stderr)
        send({"status": "startup_error", "protocol_version": _PROTOCOL_VERSION})
        return 1

    backend = ChatterboxBackend(
        device=args.device,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
    )
    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend.load()
    except Exception:
        log_failure("The isolated Chatterbox worker failed during startup.")
        send({"status": "startup_error", "protocol_version": _PROTOCOL_VERSION})
        return 1

    send({"status": "ready", "protocol_version": _PROTOCOL_VERSION})
    for line in sys.stdin:
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Worker request must be a JSON object.")
            request_id = request.get("id")
            if not isinstance(request_id, int) or isinstance(request_id, bool):
                raise ValueError("Worker request ID must be an integer.")

            action = request.get("action")
            if action == "shutdown":
                with contextlib.redirect_stdout(sys.stderr):
                    backend.unload()
                send({"id": request_id, "status": "stopped"})
                return 0
            text = request.get("text")
            if (
                action != "synthesize"
                or not isinstance(text, str)
                or not text
                or len(text) > args.max_chunk_length
            ):
                raise ValueError("Invalid Chatterbox synthesis request.")

            with contextlib.redirect_stdout(sys.stderr):
                wav_bytes = backend.synthesize(text)
            send(
                {
                    "id": request_id,
                    "status": "ok",
                    "audio": base64.b64encode(wav_bytes).decode("ascii"),
                }
            )
        except Exception:
            # Never attempt another CUDA operation after an inference error.
            # The supervisor will also terminate/reap this process.
            log_failure("The isolated Chatterbox worker failed during inference.")
            try:
                send(
                    {
                        "id": request_id,
                        "status": "error",
                        "error": "inference_failed",
                    }
                )
            except Exception:
                pass
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
