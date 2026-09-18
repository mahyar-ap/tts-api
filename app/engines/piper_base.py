"""Shared implementation for Piper ONNX voices."""

from __future__ import annotations

import gc
import io
import wave
from pathlib import Path
from typing import Any

from app.engines.base import BaseEngine, download_hf_file


class PiperEngine(BaseEngine):
    """Download, load, and run one Piper voice."""

    def __init__(
        self,
        *,
        repo_id: str,
        model_filename: str,
        config_filename: str,
        cache_dir: Path,
        hf_token: str | None,
    ) -> None:
        self.repo_id = repo_id
        self.model_filename = model_filename
        self.config_filename = config_filename
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self._voice: Any | None = None

    def load(self) -> None:
        """Download the ONNX pair and initialize PiperVoice."""

        if self._voice is not None:
            return
        from piper import PiperVoice

        model_path = download_hf_file(
            repo_id=self.repo_id,
            filename=self.model_filename,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        config_path = download_hf_file(
            repo_id=self.repo_id,
            filename=self.config_filename,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        self._voice = PiperVoice.load(
            str(model_path),
            config_path=str(config_path),
        )

    def synthesize(self, text: str) -> bytes:
        """Synthesize text directly into an in-memory WAV container."""

        if self._voice is None:
            raise RuntimeError("The Piper engine is not loaded.")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            if hasattr(self._voice, "synthesize_wav"):
                self._voice.synthesize_wav(text, wav_file)
            else:
                # Piper releases before 1.3 exposed the same operation as synthesize.
                self._voice.synthesize(text, wav_file)
        return buffer.getvalue()

    def unload(self) -> None:
        """Release the ONNX runtime session."""

        self._voice = None
        gc.collect()
