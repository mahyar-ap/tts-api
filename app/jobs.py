"""Persistent job metadata and audio storage."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.schemas import JobStatus


class JobRecord(BaseModel):
    """Internal job record persisted as a JSON sidecar."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    model_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class JobStore:
    """Keep job state in memory and mirror it to persistent storage."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._records: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load valid job sidecars and reconcile interrupted jobs."""

        await asyncio.to_thread(self.output_dir.mkdir, parents=True, exist_ok=True)
        records = await asyncio.to_thread(self._load_from_disk)
        async with self._lock:
            self._records = records

        interrupted = [
            record.job_id
            for record in records.values()
            if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}
        ]
        for job_id in interrupted:
            await self.update(
                job_id,
                status=JobStatus.FAILED,
                error="Job was interrupted by a service restart.",
            )

    def _load_from_disk(self) -> dict[str, JobRecord]:
        records: dict[str, JobRecord] = {}
        for path in self.output_dir.glob("*.json"):
            try:
                record = JobRecord.model_validate_json(path.read_text("utf-8"))
                UUID(record.job_id)
            except (OSError, ValueError):
                continue

            if (
                record.status == JobStatus.SUCCEEDED
                and not self.audio_path(record.job_id).is_file()
            ):
                record = record.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "updated_at": datetime.now(UTC),
                        "error": "The generated audio file is missing.",
                    }
                )
                self._write_record(record)
            records[record.job_id] = record
        return records

    async def create(self, job_id: str, model_id: str) -> JobRecord:
        """Create and persist a queued job."""

        now = datetime.now(UTC)
        record = JobRecord(
            job_id=job_id,
            model_id=model_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            self._records[job_id] = record
            await asyncio.to_thread(self._write_record, record)
        return record.model_copy()

    async def get(self, job_id: str) -> JobRecord | None:
        """Return a defensive copy of a job record."""

        async with self._lock:
            record = self._records.get(job_id)
            return record.model_copy() if record is not None else None

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: str | None = None,
    ) -> JobRecord:
        """Update a job state and persist it atomically."""

        async with self._lock:
            current = self._records[job_id]
            record = current.model_copy(
                update={
                    "status": status,
                    "updated_at": datetime.now(UTC),
                    "error": error,
                }
            )
            self._records[job_id] = record
            await asyncio.to_thread(self._write_record, record)
            return record.model_copy()

    async def remove(self, job_id: str) -> None:
        """Remove a job that could not be admitted to the queue."""

        async with self._lock:
            self._records.pop(job_id, None)
            await asyncio.to_thread(self.metadata_path(job_id).unlink, missing_ok=True)

    async def write_audio(self, job_id: str, wav_bytes: bytes) -> Path:
        """Write WAV bytes atomically and return the final path."""

        if not wav_bytes:
            raise ValueError("The synthesis engine returned empty audio.")
        return await asyncio.to_thread(self._write_audio, job_id, wav_bytes)

    async def cleanup_expired(
        self,
        ttl_seconds: int,
        *,
        now: datetime | None = None,
    ) -> int:
        """Delete terminal jobs whose last update is older than the TTL."""

        current_time = now or datetime.now(UTC)
        cutoff = current_time - timedelta(seconds=ttl_seconds)
        terminal_states = {JobStatus.SUCCEEDED, JobStatus.FAILED}
        async with self._lock:
            expired = [
                job_id
                for job_id, record in self._records.items()
                if record.status in terminal_states and record.updated_at <= cutoff
            ]
            for job_id in expired:
                self._records.pop(job_id, None)
                await asyncio.to_thread(self._delete_job_files, job_id)
        return len(expired)

    def _write_audio(self, job_id: str, wav_bytes: bytes) -> Path:
        destination = self.audio_path(job_id)
        temporary = destination.with_suffix(".wav.tmp")
        temporary.write_bytes(wav_bytes)
        os.replace(temporary, destination)
        return destination

    def _write_record(self, record: JobRecord) -> None:
        destination = self.metadata_path(record.job_id)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _delete_job_files(self, job_id: str) -> None:
        self.audio_path(job_id).unlink(missing_ok=True)
        self.metadata_path(job_id).unlink(missing_ok=True)

    def audio_path(self, job_id: str) -> Path:
        """Return the safe WAV path for a UUID job identifier."""

        UUID(job_id)
        return self.output_dir / f"{job_id}.wav"

    def metadata_path(self, job_id: str) -> Path:
        """Return the safe metadata path for a UUID job identifier."""

        UUID(job_id)
        return self.output_dir / f"{job_id}.json"
