"""Unit tests for normalization, audio encoding, and registry locking."""

from __future__ import annotations

import asyncio
import io
import threading
import time
import wave

import pytest

from app.engines.base import (
    BaseEngine,
    concatenate_wav_bytes,
    float_audio_to_wav_bytes,
)
from app.engines.manatts_tacotron2 import normalize_persian
from app.registry import EngineRegistry, EngineSpec


def test_normalize_persian_removes_unsupported_characters() -> None:
    text = "  كِتاب\u200c ئك، إمرأة ۱۲۳ 😊  "
    assert normalize_persian(text) == "کتاب یک، امراه"


def test_float_audio_conversion_creates_pcm_wav() -> None:
    wav = float_audio_to_wav_bytes([0.0, 0.5, -0.5], 24000)
    assert wav.startswith(b"RIFF")
    assert b"WAVE" in wav[:16]


def test_wav_concatenation_preserves_frames_and_inserts_silence() -> None:
    first = float_audio_to_wav_bytes([0.25, -0.25], 1000)
    second = float_audio_to_wav_bytes([0.5, -0.5, 0.25], 1000)

    combined = concatenate_wav_bytes([first, second], silence_ms=100)

    with wave.open(io.BytesIO(combined), "rb") as wav_file:
        assert wav_file.getparams()[:4] == (1, 2, 1000, 105)
        frames = wav_file.readframes(wav_file.getnframes())
    assert frames[4:204] == b"\0" * 200


def test_wav_concatenation_rejects_mismatched_formats() -> None:
    first = float_audio_to_wav_bytes([0.25, -0.25], 22050)
    second = float_audio_to_wav_bytes([0.25, -0.25], 24000)

    with pytest.raises(ValueError, match="incompatible"):
        concatenate_wav_bytes([first, second])


class RecordingEngine(BaseEngine):
    """Track lifecycle and concurrent calls for registry tests."""

    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.active = 0
        self.maximum_active = 0
        self._counter_lock = threading.Lock()

    def load(self) -> None:
        self.events.append(f"load:{self.name}")

    def synthesize(self, text: str) -> bytes:
        with self._counter_lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.03)
        with self._counter_lock:
            self.active -= 1
        return text.encode()

    def unload(self) -> None:
        self.events.append(f"unload:{self.name}")


class FailingLoadEngine(RecordingEngine):
    """Engine double that always fails while loading."""

    def load(self) -> None:
        self.events.append(f"load:{self.name}")
        raise RuntimeError("Internal load detail with /private/model/path.")


@pytest.mark.asyncio
async def test_registry_serializes_calls_to_the_same_engine() -> None:
    events: list[str] = []
    engine = RecordingEngine("one", events)
    registry = EngineRegistry(
        [EngineSpec("one", lambda: engine)],
        unload_policy="none",
    )

    first, second = await asyncio.gather(
        registry.synthesize("one", "first"),
        registry.synthesize("one", "second"),
    )
    assert {first, second} == {b"first", b"second"}
    assert engine.maximum_active == 1
    assert events == ["load:one"]
    await registry.unload_all()
    assert events == ["load:one", "unload:one"]


@pytest.mark.asyncio
async def test_registry_unloads_conflicting_heavy_engine() -> None:
    events: list[str] = []
    first = RecordingEngine("first", events)
    second = RecordingEngine("second", events)
    registry = EngineRegistry(
        [
            EngineSpec("first", lambda: first, heavy=True),
            EngineSpec("second", lambda: second, heavy=True),
        ],
        unload_policy="single-heavy",
    )

    await registry.synthesize("first", "a")
    assert registry.loaded_models == ["first"]
    await registry.synthesize("second", "b")
    assert registry.loaded_models == ["second"]
    assert events == ["load:first", "unload:first", "load:second"]


@pytest.mark.asyncio
async def test_failed_engine_load_does_not_poison_registry() -> None:
    events: list[str] = []
    broken = FailingLoadEngine("broken", events)
    healthy = RecordingEngine("healthy", events)
    registry = EngineRegistry(
        [
            EngineSpec("broken", lambda: broken),
            EngineSpec("healthy", lambda: healthy),
        ],
        unload_policy="none",
    )

    with pytest.raises(RuntimeError, match="Internal load detail"):
        await registry.synthesize("broken", "first")
    assert await registry.synthesize("healthy", "second") == b"second"
    assert registry.loaded_models == ["healthy"]
    assert events == ["load:broken", "unload:broken", "load:healthy"]
