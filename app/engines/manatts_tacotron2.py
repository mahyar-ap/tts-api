"""ManaTTS adapter backed by a persistent isolated Python subprocess."""

from __future__ import annotations

import base64
import json
import logging
import os
import selectors
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.engines.base import BaseEngine
from app.engines.manatts_backend import normalize_persian

logger = logging.getLogger(__name__)
__all__ = ["ManaTTSTacotron2Engine", "normalize_persian"]


class ManaTTSTacotron2Engine(BaseEngine):
    """Proxy synchronous engine calls to the isolated ManaTTS environment."""

    def __init__(
        self,
        *,
        repo_dir: Path,
        python_executable: str,
        package_dir: Path,
        startup_timeout_seconds: int,
        inference_timeout_seconds: int,
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.python_executable = python_executable
        self.package_dir = package_dir.resolve()
        self.startup_timeout_seconds = startup_timeout_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self._process: subprocess.Popen[str] | None = None

    def load(self) -> None:
        """Start the isolated worker and wait until its models are resident."""

        if self._process is not None and self._process.poll() is None:
            return
        resolved_python = shutil.which(self.python_executable)
        if resolved_python is None:
            raise RuntimeError(
                "The configured ManaTTS Python interpreter is not available."
            )
        if not self.package_dir.is_dir():
            raise RuntimeError("The isolated ManaTTS package layer is not installed.")

        project_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONPATH"] = str(self.package_dir)
        process = subprocess.Popen(
            [
                resolved_python,
                "-S",
                "-m",
                "app.engines.manatts_worker",
                "--repo-dir",
                str(self.repo_dir),
            ],
            cwd=project_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._process = process
        try:
            response = self._read_response(self.startup_timeout_seconds)
        except Exception:
            self._stop_process()
            raise
        if response.get("status") != "ready":
            logger.error(
                "The isolated ManaTTS worker reported a startup error: %s",
                response.get("error", "No error detail was provided."),
            )
            self._stop_process()
            raise RuntimeError("The isolated ManaTTS worker failed to start.")

    def synthesize(self, text: str) -> bytes:
        """Request one WAV from the resident isolated ManaTTS worker."""

        self._send_request({"action": "synthesize", "text": text})
        response = self._read_response(self.inference_timeout_seconds)
        if response.get("status") != "ok":
            logger.error(
                "The isolated ManaTTS worker reported an inference error: %s",
                response.get("error", "No error detail was provided."),
            )
            raise RuntimeError("The isolated ManaTTS worker failed during inference.")
        encoded_audio = response.get("audio")
        if not isinstance(encoded_audio, str):
            raise RuntimeError("The isolated ManaTTS worker returned invalid audio.")
        return base64.b64decode(encoded_audio, validate=True)

    def unload(self) -> None:
        """Ask the worker to release models, then terminate the subprocess."""

        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                self._send_request({"action": "shutdown"})
                self._read_response(10)
            except Exception:
                logger.exception("The isolated ManaTTS worker did not stop cleanly.")
        self._stop_process()

    def _send_request(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("The isolated ManaTTS worker is not running.")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _read_response(self, timeout_seconds: int) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("The isolated ManaTTS worker is not running.")

        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout_seconds):
                raise TimeoutError("The isolated ManaTTS worker timed out.")
            line = process.stdout.readline()
        finally:
            selector.close()

        if not line:
            raise RuntimeError("The isolated ManaTTS worker exited unexpectedly.")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("The isolated ManaTTS worker returned invalid data.")
        return response

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
