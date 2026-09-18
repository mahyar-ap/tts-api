"""Persistence behavior for job metadata."""

from pathlib import Path

import pytest

from app.jobs import JobStore
from app.schemas import JobStatus


@pytest.mark.asyncio
async def test_job_store_recovers_completed_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    await store.load()
    job_id = "00000000-0000-0000-0000-000000000001"
    await store.create(job_id, "mana-piper")
    await store.write_audio(job_id, b"RIFFtest")
    await store.update(job_id, status=JobStatus.SUCCEEDED)

    recovered = JobStore(tmp_path)
    await recovered.load()
    record = await recovered.get(job_id)
    assert record is not None
    assert record.status == JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_job_store_marks_interrupted_jobs_failed(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    await store.load()
    job_id = "00000000-0000-0000-0000-000000000002"
    await store.create(job_id, "chatterbox")

    recovered = JobStore(tmp_path)
    await recovered.load()
    record = await recovered.get(job_id)
    assert record is not None
    assert record.status == JobStatus.FAILED
    assert record.error == "Job was interrupted by a service restart."


@pytest.mark.asyncio
async def test_job_store_cleanup_removes_only_terminal_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    await store.load()
    completed_id = "00000000-0000-0000-0000-000000000003"
    queued_id = "00000000-0000-0000-0000-000000000004"
    await store.create(completed_id, "mana-piper")
    await store.write_audio(completed_id, b"RIFFtest")
    await store.update(completed_id, status=JobStatus.SUCCEEDED)
    await store.create(queued_id, "chatterbox")

    removed = await store.cleanup_expired(0)
    assert removed == 1
    assert await store.get(completed_id) is None
    assert not (tmp_path / f"{completed_id}.wav").exists()
    assert not (tmp_path / f"{completed_id}.json").exists()
    assert await store.get(queued_id) is not None
