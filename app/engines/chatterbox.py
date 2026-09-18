"""Supervised proxy for the isolated Persian Chatterbox worker."""

from __future__ import annotations

import base64
import json
import logging
import os
import selectors
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.engines.base import (
    BaseEngine,
    concatenate_wav_bytes,
    validate_wav_bytes,
)
from app.text_chunks import split_sentences_for_stream

logger = logging.getLogger(__name__)
_PROTOCOL_VERSION = 1
_CHUNK_SILENCE_MS = 120

# Keep the historical name used by tests and callers.
split_text_for_chatterbox = split_sentences_for_stream


class ChatterboxEngine(BaseEngine):
    """Proxy all Chatterbox and CUDA work to one persistent subprocess."""

    def __init__(
        self,
        *,
        device: str,
        cache_dir: Path,
        hf_token: str | None,
        chunk_size: int = 300,
        startup_timeout_seconds: int = 300,
        inference_timeout_seconds: int = 1800,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir.resolve()
        self.hf_token = hf_token
        self.chunk_size = chunk_size
        self.startup_timeout_seconds = startup_timeout_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0

    def load(self) -> None:
        """Start an isolated worker and wait until Chatterbox is resident."""

        if self._process is not None and self._process.poll() is None:
            return
        self._stop_process()
        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN is required to access the gated Chatterbox repository."
            )

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_TOKEN"] = self.hf_token
        project_root = Path(__file__).resolve().parents[2]
        process = subprocess.Popen(
            self._worker_command(),
            cwd=project_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        self._process = process
        try:
            response = self._read_response(self.startup_timeout_seconds)
            if response != {
                "status": "ready",
                "protocol_version": _PROTOCOL_VERSION,
            }:
                raise RuntimeError(
                    "The isolated Chatterbox worker returned an invalid "
                    "startup response."
                )
        except Exception:
            self._stop_process()
            raise
        logger.info("The isolated Chatterbox worker is ready (pid=%d).", process.pid)

    def synthesize(self, text: str) -> bytes:
        """Generate bounded chunks sequentially and concatenate their WAV audio."""

        try:
            return concatenate_wav_bytes(
                (wav for _text, wav in self.iter_synthesize(text)),
                silence_ms=_CHUNK_SILENCE_MS,
            )
        except Exception:
            # A CUDA assertion can corrupt the entire context. Discard the child
            # for every failed inference or protocol exchange, not just known
            # exception strings, so the next request gets a clean CUDA process.
            self._stop_process()
            raise

    def iter_synthesize(self, text: str) -> Iterator[tuple[str, bytes]]:
        """Yield ``(chunk_text, wav_bytes)`` as each Chatterbox chunk completes."""

        if self._process is None or self._process.poll() is not None:
            self.load()

        chunks = split_text_for_chatterbox(text, self.chunk_size)
        deadline = time.monotonic() + self.inference_timeout_seconds
        try:
            yield from zip(
                chunks,
                self._synthesize_chunks(chunks, deadline),
                strict=True,
            )
        except Exception:
            self._stop_process()
            raise

    def _synthesize_chunks(
        self,
        chunks: list[str],
        deadline: float,
    ) -> Iterator[bytes]:
        for chunk in chunks:
            self._request_id += 1
            request_id = self._request_id
            self._send_request(
                {
                    "id": request_id,
                    "action": "synthesize",
                    "text": chunk,
                }
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The isolated Chatterbox worker timed out.")
            response = self._read_response(remaining)
            if response.get("id") != request_id:
                raise RuntimeError(
                    "The isolated Chatterbox worker returned a mismatched response."
                )
            if response.get("status") != "ok":
                raise RuntimeError(
                    "The isolated Chatterbox worker failed during inference."
                )
            encoded_audio = response.get("audio")
            if not isinstance(encoded_audio, str):
                raise RuntimeError(
                    "The isolated Chatterbox worker returned invalid audio."
                )
            wav_bytes = base64.b64decode(encoded_audio, validate=True)
            validate_wav_bytes(wav_bytes)
            yield wav_bytes

    def unload(self) -> None:
        """Request a clean shutdown and then reap the worker process."""

        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request_id += 1
                request_id = self._request_id
                self._send_request({"id": request_id, "action": "shutdown"})
                response = self._read_response(10)
                if response != {"id": request_id, "status": "stopped"}:
                    raise RuntimeError(
                        "The isolated Chatterbox worker returned an invalid "
                        "shutdown response."
                    )
            except Exception:
                logger.exception(
                    "The isolated Chatterbox worker did not stop cleanly."
                )
        self._stop_process()

    def _worker_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "app.engines.chatterbox_worker",
            "--device",
            self.device,
            "--cache-dir",
            str(self.cache_dir),
            "--max-chunk-length",
            str(self.chunk_size),
        ]

    def _send_request(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("The isolated Chatterbox worker is not running.")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _read_response(self, timeout_seconds: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("The isolated Chatterbox worker is not running.")

        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout_seconds):
                raise TimeoutError("The isolated Chatterbox worker timed out.")
            line = process.stdout.readline()
        finally:
            selector.close()

        if not line:
            raise RuntimeError("The isolated Chatterbox worker exited unexpectedly.")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "The isolated Chatterbox worker returned invalid JSON."
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError("The isolated Chatterbox worker returned invalid data.")
        return response

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
