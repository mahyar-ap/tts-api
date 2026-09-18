"""API integration tests using a lightweight fake engine registry."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.engines.base import float_audio_to_wav_bytes
from app.main import create_app
from app.queue import PUBLIC_JOB_FAILURE

MODEL_IDS = [
    "chatterbox",
    "manatts-tacotron2",
    "mana-piper",
    "piper-ganji",
    "piper-ganji-adabi",
    "khadijah-matcha",
]


class FakeRegistry:
    """Minimal registry double that can block or fail synthesis."""

    def __init__(self) -> None:
        self.model_ids = MODEL_IDS
        self.loaded_models: list[str] = []
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.unloaded = False

    async def synthesize(self, model_id: str, text: str) -> bytes:
        self.loaded_models = [model_id]
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        if text == "trigger failure":
            raise RuntimeError("Synthetic engine failure at /private/model/path.")
        return float_audio_to_wav_bytes([0.0, 0.25, -0.25, 0.0], 22050)

    async def unload_all(self) -> None:
        self.loaded_models = []
        self.unloaded = True


def make_client(
    output_dir: Path,
    registry: FakeRegistry,
    *,
    max_queue_size: int = 100,
    **settings_overrides: object,
) -> TestClient:
    values: dict[str, object] = {
        "output_dir": output_dir,
        "model_cache_dir": output_dir / "cache",
        "max_queue_size": max_queue_size,
        "worker_count": 1,
    }
    values.update(settings_overrides)
    settings = Settings(**values)
    return TestClient(create_app(settings=settings, registry=registry))


def wait_for_terminal_status(
    client: TestClient,
    status_url: str,
    timeout: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("The job did not reach a terminal state.")


def test_service_discovery_and_documentation(tmp_path: Path) -> None:
    registry = FakeRegistry()
    with make_client(tmp_path / "outputs", registry) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "queue_size": 0,
            "queue_capacity": 100,
            "worker_count": 1,
            "loaded_models": [],
        }

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert [item["id"] for item in models.json()["models"]] == MODEL_IDS
        assert all(
            item["max_text_length"] == 10000 for item in models.json()["models"]
        )

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        text_schema = openapi.json()["components"]["schemas"]["TTSRequest"][
            "properties"
        ]["text"]
        assert text_schema["maxLength"] == 10000

        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200

    assert registry.unloaded is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   \n\t  "},
        {"text": "x" * 10001},
        {"text": "valid", "unexpected": True},
    ],
)
def test_text_validation_returns_422(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    with make_client(tmp_path / "outputs", FakeRegistry()) as client:
        response = client.post("/v1/tts/mana-piper", json=payload)
    assert response.status_code == 422


def test_successful_job_persists_and_serves_wav(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    with make_client(output_dir, FakeRegistry()) as client:
        response = client.post(
            "/v1/tts/piper-ganji",
            json={"text": "  test request  "},
        )
        assert response.status_code == 202
        created = response.json()
        assert created["status"] == "queued"
        assert created["job_id"] in created["status_url"]
        assert created["job_id"] in created["audio_url"]

        job = wait_for_terminal_status(client, created["status_url"])
        assert job["status"] == "succeeded"
        assert job["model_id"] == "piper-ganji"
        assert job["error"] is None

        audio = client.get(created["audio_url"])
        assert audio.status_code == 200
        assert audio.headers["content-type"].startswith("audio/wav")
        assert audio.content.startswith(b"RIFF")

    job_id = created["job_id"]
    assert (output_dir / f"{job_id}.json").is_file()
    assert (output_dir / f"{job_id}.wav").is_file()


def test_ten_thousand_character_request_is_accepted(tmp_path: Path) -> None:
    with make_client(tmp_path / "outputs", FakeRegistry()) as client:
        response = client.post(
            "/v1/tts/chatterbox",
            json={"text": "س" * 10000},
        )
        assert response.status_code == 202
        job = wait_for_terminal_status(client, response.json()["status_url"])
        assert job["status"] == "succeeded"


def test_max_text_length_environment_accepts_ten_thousand(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAX_TEXT_LENGTH", "10000")
    assert Settings(_env_file=None).max_text_length == 10000
    monkeypatch.setenv("MAX_TEXT_LENGTH", "10001")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_failed_job_exposes_error_and_audio_returns_500(tmp_path: Path) -> None:
    with make_client(tmp_path / "outputs", FakeRegistry()) as client:
        response = client.post(
            "/v1/tts/chatterbox",
            json={"text": "trigger failure"},
        )
        created = response.json()
        job = wait_for_terminal_status(client, created["status_url"])
        assert job["status"] == "failed"
        assert job["error"] == PUBLIC_JOB_FAILURE

        audio = client.get(created["audio_url"])
        assert audio.status_code == 500
        assert audio.json() == {"detail": PUBLIC_JOB_FAILURE}

        second = client.post(
            "/v1/tts/chatterbox",
            json={"text": "a successful request"},
        )
        assert second.status_code == 202
        assert (
            wait_for_terminal_status(client, second.json()["status_url"])["status"]
            == "succeeded"
        )


def test_not_ready_and_full_queue_responses(tmp_path: Path) -> None:
    registry = FakeRegistry()
    registry.release.clear()
    with make_client(
        tmp_path / "outputs",
        registry,
        max_queue_size=1,
    ) as client:
        first = client.post("/v1/tts/mana-piper", json={"text": "first"})
        assert first.status_code == 202
        assert registry.started.wait(timeout=1.0)

        pending_audio = client.get(first.json()["audio_url"])
        assert pending_audio.status_code == 409

        second = client.post("/v1/tts/mana-piper", json={"text": "second"})
        assert second.status_code == 202
        rejected = client.post("/v1/tts/mana-piper", json={"text": "third"})
        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "5"
        assert rejected.json() == {"detail": "The synthesis queue is full."}

        registry.release.set()
        assert (
            wait_for_terminal_status(client, first.json()["status_url"])["status"]
            == "succeeded"
        )
        assert (
            wait_for_terminal_status(client, second.json()["status_url"])["status"]
            == "succeeded"
        )


def test_unknown_job_returns_404(tmp_path: Path) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    with make_client(tmp_path / "outputs", FakeRegistry()) as client:
        assert client.get(f"/v1/jobs/{missing}").status_code == 404
        assert client.get(f"/v1/jobs/{missing}/audio").status_code == 404


def test_optional_api_key_protects_versioned_routes(tmp_path: Path) -> None:
    with make_client(
        tmp_path / "outputs",
        FakeRegistry(),
        api_key_enabled=True,
        api_key="test-secret",
        api_key_header="X-Test-Key",
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models",
                headers={"X-Test-Key": "wrong"},
            ).status_code
            == 401
        )
        authorized = client.get(
            "/v1/models",
            headers={"X-Test-Key": "test-secret"},
        )
        assert authorized.status_code == 200
