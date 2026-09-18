#!/usr/bin/env python3
"""Small protocol-compatible Chatterbox worker used by supervisor tests."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import wave
from pathlib import Path


def send(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def record(event: dict[str, object]) -> None:
    path_value = os.environ.get("FAKE_CHATTERBOX_EVENT_FILE")
    if path_value:
        with Path(path_value).open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x01\x00\x02\x00")
    return output.getvalue()


pid = os.getpid()
record({"event": "startup", "pid": pid})
startup_failure = os.environ.get("FAKE_CHATTERBOX_STARTUP_FAILURE")
if startup_failure == "error":
    send({"status": "startup_error", "protocol_version": 1})
    raise SystemExit(1)
if startup_failure == "crash":
    os._exit(24)
if startup_failure == "protocol":
    sys.stdout.write("not-json\n")
    sys.stdout.flush()
    raise SystemExit(1)
if startup_failure == "timeout":
    time.sleep(10)
send({"status": "ready", "protocol_version": 1})
for line in sys.stdin:
    request = json.loads(line)
    action = request.get("action")
    if action == "shutdown":
        record({"event": "shutdown", "pid": pid})
        send({"id": request["id"], "status": "stopped"})
        raise SystemExit(0)

    text = request.get("text")
    record({"event": "synthesize", "pid": pid, "text": text})
    if text == "__crash__":
        os._exit(23)
    if text == "__error__":
        send({"id": request["id"], "status": "error"})
        raise SystemExit(1)
    if text == "__protocol__":
        sys.stdout.write("not-json\n")
        sys.stdout.flush()
        raise SystemExit(1)
    if text == "__timeout__":
        time.sleep(10)

    send(
        {
            "id": request["id"],
            "status": "ok",
            "audio": base64.b64encode(wav_bytes()).decode("ascii"),
        }
    )
