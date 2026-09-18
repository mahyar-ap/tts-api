"""Common engine interface and audio conversion helpers."""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


class BaseEngine(ABC):
    """Synchronous engine contract managed by the asynchronous registry."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize the inference runtime."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text and return a complete WAV file."""

    @abstractmethod
    def unload(self) -> None:
        """Release model references and accelerator memory."""


def download_hf_file(
    *,
    repo_id: str,
    filename: str,
    cache_dir: Path,
    token: str | None,
    revision: str | None = None,
) -> Path:
    """Download a Hugging Face file into the persistent model cache."""

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir),
        token=token,
    )
    return Path(path)


def float_audio_to_wav_bytes(
    samples: Any,
    sample_rate: int,
    *,
    normalize: bool = False,
) -> bytes:
    """Encode floating-point mono samples as a 16-bit PCM WAV file."""

    array = np.asarray(samples, dtype=np.float32).squeeze()
    if array.ndim != 1 or array.size == 0:
        raise ValueError("The engine returned an invalid audio shape.")
    if sample_rate <= 0:
        raise ValueError("The engine returned an invalid sample rate.")

    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(array)))
    if normalize and peak > 0:
        array = array / peak * 0.97
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype("<i2", copy=False).tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def validate_wav_bytes(wav_bytes: bytes) -> None:
    """Raise ``ValueError`` unless bytes contain non-empty PCM WAV audio."""

    _read_pcm_wav(wav_bytes)


def concatenate_wav_bytes(
    wav_chunks: Iterable[bytes],
    *,
    silence_ms: int = 120,
) -> bytes:
    """Join compatible PCM WAV files with a short silent gap."""

    chunk_iterator = iter(wav_chunks)
    try:
        first_chunk = next(chunk_iterator)
    except StopIteration as exc:
        raise ValueError("At least one WAV chunk is required.") from exc
    if silence_ms < 0:
        raise ValueError("WAV silence duration cannot be negative.")

    first_format, first_frames = _read_pcm_wav(first_chunk)
    channels, sample_width, sample_rate = first_format
    silence_frames = round(sample_rate * silence_ms / 1000)
    silence = b"\0" * (silence_frames * channels * sample_width)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframesraw(first_frames)
        for wav_bytes in chunk_iterator:
            audio_format, frames = _read_pcm_wav(wav_bytes)
            if audio_format != first_format:
                raise ValueError("WAV chunks use incompatible audio formats.")
            wav_file.writeframesraw(silence)
            wav_file.writeframesraw(frames)
    return output.getvalue()


def _read_pcm_wav(wav_bytes: bytes) -> tuple[tuple[int, int, int], bytes]:
    if not isinstance(wav_bytes, bytes) or not wav_bytes:
        raise ValueError("The synthesis worker returned empty audio.")

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("Only uncompressed PCM WAV audio is supported.")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError("The synthesis worker returned an invalid WAV file.") from exc

    if (
        channels <= 0
        or sample_width <= 0
        or sample_rate <= 0
        or frame_count <= 0
    ):
        raise ValueError("The synthesis worker returned an invalid WAV file.")
    frame_width = channels * sample_width
    if len(frames) != frame_count * frame_width:
        raise ValueError("The synthesis worker returned a truncated WAV file.")
    return (channels, sample_width, sample_rate), frames
