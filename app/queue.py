"""Bounded asynchronous queue and background synthesis workers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4

from app.jobs import JobRecord, JobStore
from app.registry import EngineRegistry
from app.schemas import JobStatus

logger = logging.getLogger(__name__)
PUBLIC_JOB_FAILURE = "The synthesis engine failed to process this job."


class QueueCapacityError(RuntimeError):
    """Raised when a synthesis request cannot enter the bounded queue."""


@dataclass(frozen=True, slots=True)
class SynthesisJob:
    """Queue payload containing transient request text."""

    job_id: str
    model_id: str
    text: str


class JobQueue:
    """Own the bounded queue and its background workers."""

    def __init__(
        self,
        *,
        maxsize: int,
        worker_count: int,
        store: JobStore,
        registry: EngineRegistry,
    ) -> None:
        self.queue: asyncio.Queue[SynthesisJob | None] = asyncio.Queue(maxsize=maxsize)
        self.worker_count = worker_count
        self.store = store
        self.registry = registry
        self._workers: list[asyncio.Task[None]] = []
        self._accepting = False

    @property
    def size(self) -> int:
        """Return the number of jobs waiting for a worker."""

        return self.queue.qsize()

    @property
    def capacity(self) -> int:
        """Return the configured queue capacity."""

        return self.queue.maxsize

    async def start(self) -> None:
        """Start all configured worker tasks."""

        if self._workers:
            return
        self._accepting = True
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"tts-worker-{index}")
            for index in range(self.worker_count)
        ]
        logger.info("Started %d synthesis worker(s).", self.worker_count)

    async def submit(self, model_id: str, text: str) -> JobRecord:
        """Persist and enqueue a new job without waiting for capacity."""

        if not self._accepting:
            raise RuntimeError("The synthesis queue is not accepting jobs.")

        job_id = str(uuid4())
        record = await self.store.create(job_id, model_id)
        try:
            self.queue.put_nowait(SynthesisJob(job_id, model_id, text))
        except asyncio.QueueFull as exc:
            await self.store.remove(job_id)
            raise QueueCapacityError("The synthesis queue is full.") from exc
        return record

    async def shutdown(self) -> None:
        """Stop admission, finish accepted jobs, and terminate workers."""

        self._accepting = False
        if not self._workers:
            return

        await self.queue.join()
        for _ in self._workers:
            await self.queue.put(None)
        await asyncio.gather(*self._workers)
        self._workers.clear()
        logger.info("Stopped all synthesis workers.")

    async def _worker(self, index: int) -> None:
        logger.info("Synthesis worker %d is ready.", index)
        while True:
            job = await self.queue.get()
            try:
                if job is None:
                    return
                try:
                    await self._process(job)
                except Exception:
                    # Storage failures are handled here so one damaged job cannot
                    # permanently reduce the configured worker pool.
                    logger.exception(
                        "Synthesis worker %d could not finalize job %s.",
                        index,
                        job.job_id,
                    )
            finally:
                self.queue.task_done()

    async def _process(self, job: SynthesisJob) -> None:
        await self.store.update(job.job_id, status=JobStatus.RUNNING)
        logger.info("Synthesis job %s started with model %s.", job.job_id, job.model_id)
        try:
            wav_bytes = await self.registry.synthesize(job.model_id, job.text)
            await self.store.write_audio(job.job_id, wav_bytes)
            await self.store.update(job.job_id, status=JobStatus.SUCCEEDED)
        except Exception:  # The worker must survive third-party failures.
            await self.store.update(
                job.job_id,
                status=JobStatus.FAILED,
                error=PUBLIC_JOB_FAILURE,
            )
            logger.exception(
                "Synthesis job %s failed for model %s.",
                job.job_id,
                job.model_id,
            )
        else:
            logger.info("Synthesis job %s completed successfully.", job.job_id)
