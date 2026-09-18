"""Tests for Chatterbox isolation, chunking, and checkpoint integration."""

from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from app.engines import chatterbox_worker
from app.engines.chatterbox import ChatterboxEngine, split_text_for_chatterbox
from app.engines.chatterbox_backend import ChatterboxBackend

FAKE_WORKER = Path(__file__).parent / "fixtures" / "fake_chatterbox_worker.py"


class FixtureWorkerEngine(ChatterboxEngine):
    """Run the real supervisor against a lightweight external process."""

    def _worker_command(self) -> list[str]:
        return [sys.executable, str(FAKE_WORKER)]


def make_fixture_engine(
    tmp_path: Path,
    *,
    chunk_size: int = 300,
    startup_timeout_seconds: float = 2,
    inference_timeout_seconds: float = 2,
) -> FixtureWorkerEngine:
    return FixtureWorkerEngine(
        device="cuda",
        cache_dir=tmp_path / "cache",
        hf_token="test-token",
        chunk_size=chunk_size,
        startup_timeout_seconds=startup_timeout_seconds,
        inference_timeout_seconds=inference_timeout_seconds,
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_worker_starts_once_reuses_model_and_shuts_down(
    monkeypatch,
    tmp_path: Path,
) -> None:
    event_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_CHATTERBOX_EVENT_FILE", str(event_file))
    engine = make_fixture_engine(tmp_path)

    engine.load()
    assert engine._process is not None
    first_pid = engine._process.pid
    engine.load()
    assert engine.synthesize("درخواست اول").startswith(b"RIFF")
    assert engine.synthesize("درخواست دوم").startswith(b"RIFF")
    assert engine._process is not None
    assert engine._process.pid == first_pid
    engine.unload()

    events = read_events(event_file)
    assert [event["event"] for event in events] == [
        "startup",
        "synthesize",
        "synthesize",
        "shutdown",
    ]
    assert {event["pid"] for event in events} == {first_pid}
    assert engine._process is None


@pytest.mark.parametrize(
    ("failure", "expected_exception", "timeout"),
    [
        ("error", RuntimeError, 2),
        ("crash", RuntimeError, 2),
        ("protocol", RuntimeError, 2),
        ("timeout", TimeoutError, 0.1),
    ],
)
def test_startup_failure_terminates_worker(
    monkeypatch,
    tmp_path: Path,
    failure: str,
    expected_exception: type[Exception],
    timeout: float,
) -> None:
    monkeypatch.setenv("FAKE_CHATTERBOX_STARTUP_FAILURE", failure)
    engine = make_fixture_engine(
        tmp_path,
        startup_timeout_seconds=timeout,
    )

    with pytest.raises(expected_exception):
        engine.load()
    assert engine._process is None


@pytest.mark.parametrize(
    ("trigger", "expected_exception", "timeout"),
    [
        ("__error__", RuntimeError, 2),
        ("__crash__", RuntimeError, 2),
        ("__protocol__", RuntimeError, 2),
        ("__timeout__", TimeoutError, 0.1),
    ],
)
def test_failed_worker_is_replaced_on_next_request(
    monkeypatch,
    tmp_path: Path,
    trigger: str,
    expected_exception: type[Exception],
    timeout: float,
) -> None:
    event_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_CHATTERBOX_EVENT_FILE", str(event_file))
    engine = make_fixture_engine(
        tmp_path,
        inference_timeout_seconds=timeout,
    )
    engine.load()
    assert engine._process is not None
    failed_pid = engine._process.pid

    with pytest.raises(expected_exception):
        engine.synthesize(trigger)
    assert engine._process is None

    assert engine.synthesize("درخواست سالم").startswith(b"RIFF")
    assert engine._process is not None
    replacement_pid = engine._process.pid
    assert replacement_pid != failed_pid
    engine.unload()

    startup_pids = [
        event["pid"]
        for event in read_events(event_file)
        if event["event"] == "startup"
    ]
    assert startup_pids == [failed_pid, replacement_pid]


def test_long_text_is_sent_as_sequential_bounded_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    event_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_CHATTERBOX_EVENT_FILE", str(event_file))
    engine = make_fixture_engine(tmp_path, chunk_size=45)
    text = (
        "این جمله اول است. این جمله دوم کمی بلندتر است؟ "
        + "واژه " * 24
    ).strip()
    expected_chunks = split_text_for_chatterbox(text, 45)

    output = engine.synthesize(text)
    engine.unload()

    synthesized = [
        event["text"]
        for event in read_events(event_file)
        if event["event"] == "synthesize"
    ]
    assert synthesized == expected_chunks
    assert len(synthesized) > 1
    assert all(0 < len(chunk) <= 45 for chunk in synthesized)

    with wave.open(io.BytesIO(output), "rb") as wav_file:
        expected_frames = 2 * len(synthesized) + 2880 * (len(synthesized) - 1)
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == expected_frames


def test_chunking_prefers_persian_and_latin_sentence_boundaries() -> None:
    text = (
        "جمله فارسی اول تمام شد. جمله دوم هم تمام شد؟ "
        "A Latin sentence ends here! "
        "این جمله بسیار بلند است، و باید در یک مرز طبیعی شکسته شود "
        "تا هیچ فراخوانی از حد تعیین شده عبور نکند."
    )
    chunks = split_text_for_chatterbox(text, 55)

    assert all(len(chunk) <= 55 for chunk in chunks)
    assert "." in chunks[0]
    assert chunks[0].endswith("؟")
    assert chunks[1].endswith("!")
    assert " ".join(chunks) == " ".join(text.split())


def test_chunking_recognizes_boundaries_without_following_spaces() -> None:
    chunks = split_text_for_chatterbox(
        "اول تمام شد؟»دوم تمام شد!Third sentence.",
        20,
    )

    assert chunks == ["اول تمام شد؟»", "دوم تمام شد!", "Third sentence."]


def test_real_worker_protocol_runs_startup_inference_and_shutdown(
    monkeypatch,
) -> None:
    events: list[object] = []
    wav = _small_wav()

    class FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            events.append(("init", kwargs))

        def load(self) -> None:
            print("backend startup output")
            events.append("load")

        def synthesize(self, text: str) -> bytes:
            print("backend inference output")
            events.append(("synthesize", text))
            return wav

        def unload(self) -> None:
            print("backend shutdown output")
            events.append("unload")

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(chatterbox_worker, "ChatterboxBackend", FakeBackend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chatterbox_worker",
            "--device",
            "cuda",
            "--cache-dir",
            "/cache",
            "--max-chunk-length",
            "300",
        ],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"id": 1, "action": "synthesize", "text": "سلام"})
            + "\n"
            + json.dumps({"id": 2, "action": "shutdown"})
            + "\n"
        ),
    )
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setenv("HF_TOKEN", "worker-token")

    assert chatterbox_worker.main() == 0
    messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert messages[0] == {"status": "ready", "protocol_version": 1}
    assert messages[1]["id"] == 1
    assert messages[1]["status"] == "ok"
    assert messages[2] == {"id": 2, "status": "stopped"}
    assert events[1:] == ["load", ("synthesize", "سلام"), "unload"]
    assert stderr.getvalue().splitlines() == [
        "backend startup output",
        "backend inference output",
        "backend shutdown output",
    ]


def test_real_worker_reports_startup_failure(monkeypatch) -> None:
    class FailingBackend:
        def __init__(self, **kwargs: object) -> None:
            pass

        def load(self) -> None:
            raise RuntimeError("CUDA initialization failed")

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(chatterbox_worker, "ChatterboxBackend", FailingBackend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chatterbox_worker",
            "--device",
            "cuda",
            "--cache-dir",
            "/cache",
            "--max-chunk-length",
            "300",
        ],
    )
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setenv("HF_TOKEN", "worker-token")

    assert chatterbox_worker.main() == 1
    assert json.loads(stdout.getvalue()) == {
        "status": "startup_error",
        "protocol_version": 1,
    }
    assert "failed during startup" in stderr.getvalue()
    assert "CUDA initialization failed" in stderr.getvalue()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("CUDA error"),
        RuntimeError("CUDA device-side assert triggered"),
        IndexError("token index out of range"),
    ],
)
def test_real_worker_exits_after_generation_failure(
    monkeypatch,
    failure: Exception,
) -> None:
    class FailingBackend:
        def __init__(self, **kwargs: object) -> None:
            pass

        def load(self) -> None:
            pass

        def synthesize(self, text: str) -> bytes:
            raise failure

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(chatterbox_worker, "ChatterboxBackend", FailingBackend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chatterbox_worker",
            "--device",
            "cuda",
            "--cache-dir",
            "/cache",
            "--max-chunk-length",
            "300",
        ],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"id": 9, "action": "synthesize", "text": "bad"})
            + "\n"
        ),
    )
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setenv("HF_TOKEN", "worker-token")

    assert chatterbox_worker.main() == 1
    messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert messages[-1] == {
        "id": 9,
        "status": "error",
        "error": "inference_failed",
    }
    assert "failed during inference" in stderr.getvalue()
    assert str(failure) in stderr.getvalue()


def test_persian_checkpoint_is_loaded_strictly_into_t3(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[Any, ...]] = []
    inference_active = False

    class InferenceMode:
        def __enter__(self) -> None:
            nonlocal inference_active
            inference_active = True
            events.append(("inference-enter",))

        def __exit__(self, *args: object) -> None:
            nonlocal inference_active
            inference_active = False
            events.append(("inference-exit",))

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            events.append(("empty-cache",))

    torch_module = ModuleType("torch")
    torch_module.inference_mode = InferenceMode
    torch_module.cuda = FakeCuda()

    class FakeT3:
        def load_state_dict(self, state_dict: object, *, strict: bool) -> None:
            events.append(("load-state", state_dict, strict))

        def to(self, device: str) -> FakeT3:
            events.append(("t3-to", device))
            return self

        def eval(self) -> FakeT3:
            events.append(("t3-eval",))
            return self

    class FakeModel:
        sr = 24000

        def __init__(self) -> None:
            self.t3 = FakeT3()

        def generate(self, **kwargs: object) -> list[float]:
            assert inference_active is True
            events.append(("generate", kwargs))
            return [0.0, 0.25, -0.25]

    fake_model = FakeModel()

    class FakeChatterboxMultilingualTTS:
        @staticmethod
        def from_pretrained(*, device: str) -> FakeModel:
            events.append(("from-pretrained", device))
            return fake_model

    chatterbox_package = ModuleType("chatterbox")
    chatterbox_module = ModuleType("chatterbox.mtl_tts")
    chatterbox_module.ChatterboxMultilingualTTS = FakeChatterboxMultilingualTTS

    checkpoint = tmp_path / "t3_fa.safetensors"
    checkpoint.write_bytes(b"mock")
    hub_module = ModuleType("huggingface_hub")

    def fake_hf_hub_download(**kwargs: object) -> str:
        events.append(("download", kwargs))
        return str(checkpoint)

    hub_module.hf_hub_download = fake_hf_hub_download
    safetensors_package = ModuleType("safetensors")
    safetensors_torch = ModuleType("safetensors.torch")

    def fake_load_file(path: str, *, device: str) -> dict[str, int]:
        events.append(("load-file", path, device))
        return {"weight": 1}

    safetensors_torch.load_file = fake_load_file

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "chatterbox", chatterbox_package)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", chatterbox_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_package)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    backend = ChatterboxBackend(
        device="cuda",
        cache_dir=tmp_path / "cache",
        hf_token="secret-token",
    )
    backend.load()
    backend.load()
    wav_bytes = backend.synthesize("test")
    backend.unload()

    assert wav_bytes.startswith(b"RIFF")
    assert events.count(("from-pretrained", "cuda")) == 1
    assert ("load-state", {"weight": 1}, True) in events
    assert ("t3-to", "cuda") in events
    assert ("t3-eval",) in events
    download_event = next(event for event in events if event[0] == "download")
    assert download_event[1] == {
        "repo_id": "Thomcles/Chatterbox-TTS-Persian-Farsi",
        "filename": "t3_fa.safetensors",
        "cache_dir": str(tmp_path / "cache"),
        "token": "secret-token",
    }
    generate_event = next(event for event in events if event[0] == "generate")
    assert generate_event[1] == {
        "text": "test",
        "language_id": None,
        "audio_prompt_path": None,
        "temperature": 0.7,
        "cfg_weight": 0.5,
        "top_p": 0.5,
        "exaggeration": 0.6,
    }
    assert ("inference-enter",) in events
    assert ("inference-exit",) in events
    assert ("empty-cache",) in events
    assert backend._model is None


def _small_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\0\0\1\0")
    return output.getvalue()
