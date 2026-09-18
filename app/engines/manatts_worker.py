"""Line-oriented isolated worker process for the legacy ManaTTS stack."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from app.engines.manatts_backend import ManaTTSBackend


def send(payload: dict[str, Any]) -> None:
    """Write one protocol response to standard output."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def log_failure(message: str) -> None:
    """Write an English failure heading and full traceback to standard error."""

    print(message, file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def main() -> None:
    """Load ManaTTS once and process JSON requests until shutdown."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    args = parser.parse_args()
    backend = ManaTTSBackend(repo_dir=args.repo_dir)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend.load()
    except Exception as exc:
        log_failure("The isolated ManaTTS worker failed during startup.")
        send({"status": "error", "error": str(exc)})
        return

    send({"status": "ready"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.get("action")
            if action == "shutdown":
                with contextlib.redirect_stdout(sys.stderr):
                    backend.unload()
                send({"status": "stopped"})
                return
            if action != "synthesize" or not isinstance(request.get("text"), str):
                send({"status": "error", "error": "Invalid worker request."})
                continue

            with contextlib.redirect_stdout(sys.stderr):
                wav_bytes = backend.synthesize(request["text"])
            send(
                {
                    "status": "ok",
                    "audio": base64.b64encode(wav_bytes).decode("ascii"),
                }
            )
        except Exception as exc:
            log_failure("The isolated ManaTTS worker failed during inference.")
            send({"status": "error", "error": str(exc)})


if __name__ == "__main__":
    main()
