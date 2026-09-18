"""FastAPI application for queued Persian text-to-speech synthesis."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.jobs import JobRecord, JobStore
from app.queue import JobQueue, QueueCapacityError
from app.registry import PUBLIC_SYNTHESIS_FAILURE, EngineRegistry, build_registry
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    JobCreated,
    JobResponse,
    JobStatus,
    ModelInfo,
    ModelLifecycleResponse,
    ModelsResponse,
    TTSRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    registry: EngineRegistry | None = None,
) -> FastAPI:
    """Create an application, with optional dependencies for testing."""

    runtime_settings = settings or get_settings()
    if runtime_settings.api_key_enabled and not runtime_settings.api_key:
        raise ValueError("API_KEY must be set when API_KEY_ENABLED is true.")
    runtime_registry = registry or build_registry(runtime_settings)
    store = JobStore(runtime_settings.output_dir)
    job_queue = JobQueue(
        maxsize=runtime_settings.max_queue_size,
        worker_count=runtime_settings.worker_count,
        store=store,
        registry=runtime_registry,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime_settings.output_dir.mkdir(parents=True, exist_ok=True)
        runtime_settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        await store.load()
        await job_queue.start()
        cleanup_stop = asyncio.Event()
        cleanup_task: asyncio.Task[None] | None = None
        if runtime_settings.job_ttl_seconds > 0:
            cleanup_task = asyncio.create_task(
                cleanup_expired_jobs(cleanup_stop),
                name="job-cleanup",
            )
        logger.info("Persian TTS API startup completed.")
        try:
            yield
        finally:
            logger.info("Persian TTS API shutdown started.")
            cleanup_stop.set()
            if cleanup_task is not None:
                await cleanup_task
            await job_queue.shutdown()
            await runtime_registry.unload_all()
            logger.info("Persian TTS API shutdown completed.")

    application = FastAPI(
        title="Persian Open-Weight TTS API",
        version="1.0.0",
        description=(
            "Asynchronous Persian speech synthesis backed by multiple "
            "open-weight engines, with optional SSE chunk streaming."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.registry = runtime_registry
    application.state.job_store = store
    application.state.job_queue = job_queue

    async def cleanup_expired_jobs(stop: asyncio.Event) -> None:
        """Periodically delete expired terminal jobs and their output files."""

        while not stop.is_set():
            try:
                removed = await store.cleanup_expired(runtime_settings.job_ttl_seconds)
                if removed:
                    logger.info("Deleted %d expired synthesis job(s).", removed)
            except Exception:
                logger.exception("Expired synthesis job cleanup failed.")

            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=runtime_settings.job_cleanup_interval_seconds,
                )
            except TimeoutError:
                continue

    @application.middleware("http")
    async def require_api_key(request: Request, call_next):
        """Protect versioned API routes when API-key authentication is enabled."""

        if runtime_settings.api_key_enabled and request.url.path.startswith("/v1/"):
            provided_key = request.headers.get(runtime_settings.api_key_header)
            expected_key = runtime_settings.api_key
            if (
                provided_key is None
                or expected_key is None
                or not secrets.compare_digest(provided_key, expected_key)
            ):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "A valid API key is required."},
                    headers={"WWW-Authenticate": "ApiKey"},
                )
        return await call_next(request)

    async def submit_job(
        model_id: str,
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        try:
            record = await job_queue.submit(model_id, payload.text)
        except QueueCapacityError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The synthesis queue is full.",
                headers={
                    "Retry-After": str(runtime_settings.retry_after_seconds),
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The synthesis service is not accepting jobs.",
            ) from exc
        return JobCreated(
            job_id=record.job_id,
            status=record.status,
            status_url=str(request.url_for("get_job", job_id=record.job_id)),
            audio_url=str(request.url_for("get_job_audio", job_id=record.job_id)),
        )

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream_synthesis(model_id: str, payload: TTSRequest) -> StreamingResponse:
        async def event_source() -> AsyncIterator[str]:
            count = 0
            t0 = time.perf_counter()
            t_prev = t0
            # Open the chunked response immediately so the client does not wait
            # for the first WAV to finish generating.
            yield ": connected\n\n"
            await asyncio.sleep(0)
            try:
                async for index, chunk_text, wav_bytes in runtime_registry.synthesize_stream(
                    model_id,
                    payload.text,
                    max_chunk_length=runtime_settings.chatterbox_chunk_size,
                ):
                    now = time.perf_counter()
                    count = index + 1
                    ttfb_ms = int(round((now - t0) * 1000))
                    gap_ms = int(round((now - t_prev) * 1000))
                    t_prev = now
                    line = (
                        f"TTS chunk timings: model={model_id} index={index} "
                        f"ttfb={ttfb_ms}ms gap={gap_ms}ms bytes={len(wav_bytes)} "
                        f"chars={len(chunk_text)} text={chunk_text[:40]!r}"
                    )
                    logger.info(line)
                    print(line, flush=True)
                    yield _sse(
                        "chunk",
                        {
                            "index": index,
                            "mime": "audio/wav",
                            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                            "text": chunk_text,
                            "model_id": model_id,
                            "ttfb_ms": ttfb_ms,
                            "gap_ms": gap_ms,
                        },
                    )
                    # Let Uvicorn flush this SSE event before the next generate().
                    await asyncio.sleep(0)
                total_ms = int(round((time.perf_counter() - t0) * 1000))
                done_line = (
                    f"TTS stream done: model={model_id} chunks={count} total={total_ms}ms "
                    f"chars={len(payload.text)}"
                )
                logger.info(done_line)
                print(done_line, flush=True)
                yield _sse(
                    "done",
                    {"chunks": count, "model_id": model_id, "total_ms": total_ms},
                )
            except KeyError:
                yield _sse("error", {"detail": f"Unknown model: {model_id}"})
            except Exception:
                logger.exception("Streaming synthesis failed for model %s.", model_id)
                yield _sse("error", {"detail": PUBLIC_SYNTHESIS_FAILURE})

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        """Report service, queue, worker, and model residency state."""

        return HealthResponse(
            status="ok",
            queue_size=job_queue.size,
            queue_capacity=job_queue.capacity,
            worker_count=runtime_settings.worker_count,
            loaded_models=runtime_registry.loaded_models,
        )

    @application.get(
        "/v1/models",
        response_model=ModelsResponse,
        tags=["models"],
    )
    async def list_models() -> ModelsResponse:
        """List model identifiers accepted by the synthesis endpoints."""

        return ModelsResponse(
            models=[
                ModelInfo(
                    id=model_id,
                    max_text_length=runtime_settings.max_text_length,
                )
                for model_id in runtime_registry.model_ids
            ]
        )

    @application.post(
        "/v1/models/{model_id}/load",
        response_model=ModelLifecycleResponse,
        tags=["models"],
    )
    async def load_model(model_id: str) -> ModelLifecycleResponse:
        """Load one model into memory, unloading conflicts per policy."""

        try:
            loaded = await runtime_registry.ensure_loaded(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to load model %s.", model_id)
            raise HTTPException(
                status_code=500,
                detail="The synthesis engine failed to load this model.",
            ) from exc
        return ModelLifecycleResponse(model_id=model_id, loaded_models=loaded)

    @application.post(
        "/v1/models/{model_id}/unload",
        response_model=ModelLifecycleResponse,
        tags=["models"],
    )
    async def unload_model(model_id: str) -> ModelLifecycleResponse:
        """Unload one resident model."""

        try:
            loaded = await runtime_registry.unload_one(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to unload model %s.", model_id)
            raise HTTPException(
                status_code=500,
                detail="The synthesis engine failed to unload this model.",
            ) from exc
        return ModelLifecycleResponse(model_id=model_id, loaded_models=loaded)

    common_post_responses = {
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "The bounded synthesis queue is full.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The service is shutting down.",
        },
    }

    @application.post(
        "/v1/tts/chatterbox",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        responses=common_post_responses,
        tags=["synthesis"],
    )
    async def submit_chatterbox(
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        """Submit a Chatterbox Persian synthesis job."""

        return await submit_job("chatterbox", payload, request)

    @application.post(
        "/v1/tts/chatterbox/stream",
        tags=["synthesis"],
    )
    async def stream_chatterbox(payload: TTSRequest) -> StreamingResponse:
        """Stream Chatterbox WAV chunks as Server-Sent Events."""

        return await stream_synthesis("chatterbox", payload)

    @application.post(
        "/v1/tts/manatts-tacotron2",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        responses=common_post_responses,
        tags=["synthesis"],
    )
    async def submit_manatts_tacotron2(
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        """Submit a ManaTTS Tacotron2 synthesis job."""

        return await submit_job("manatts-tacotron2", payload, request)

    @application.post(
        "/v1/tts/manatts-tacotron2/stream",
        tags=["synthesis"],
    )
    async def stream_manatts_tacotron2(payload: TTSRequest) -> StreamingResponse:
        return await stream_synthesis("manatts-tacotron2", payload)

    @application.post(
        "/v1/tts/mana-piper",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        responses=common_post_responses,
        tags=["synthesis"],
    )
    async def submit_mana_piper(
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        """Submit a Mana Persian Piper synthesis job."""

        return await submit_job("mana-piper", payload, request)

    @application.post(
        "/v1/tts/mana-piper/stream",
        tags=["synthesis"],
    )
    async def stream_mana_piper(payload: TTSRequest) -> StreamingResponse:
        return await stream_synthesis("mana-piper", payload)

    @application.post(
        "/v1/tts/piper-ganji",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        responses=common_post_responses,
        tags=["synthesis"],
    )
    async def submit_piper_ganji(
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        """Submit a Ganji Piper synthesis job."""

        return await submit_job("piper-ganji", payload, request)

    @application.post(
        "/v1/tts/piper-ganji/stream",
        tags=["synthesis"],
    )
    async def stream_piper_ganji(payload: TTSRequest) -> StreamingResponse:
        return await stream_synthesis("piper-ganji", payload)

    @application.post(
        "/v1/tts/piper-ganji-adabi",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        responses=common_post_responses,
        tags=["synthesis"],
    )
    async def submit_piper_ganji_adabi(
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        """Submit a Ganji-Adabi Piper synthesis job."""

        return await submit_job("piper-ganji-adabi", payload, request)

    @application.post(
        "/v1/tts/piper-ganji-adabi/stream",
        tags=["synthesis"],
    )
    async def stream_piper_ganji_adabi(payload: TTSRequest) -> StreamingResponse:
        return await stream_synthesis("piper-ganji-adabi", payload)

    @application.post(
        "/v1/tts/khadijah-matcha",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        responses=common_post_responses,
        tags=["synthesis"],
    )
    async def submit_khadijah_matcha(
        payload: TTSRequest,
        request: Request,
    ) -> JobCreated:
        """Submit a Khadijah Matcha synthesis job."""

        return await submit_job("khadijah-matcha", payload, request)

    @application.post(
        "/v1/tts/khadijah-matcha/stream",
        tags=["synthesis"],
    )
    async def stream_khadijah_matcha(payload: TTSRequest) -> StreamingResponse:
        return await stream_synthesis("khadijah-matcha", payload)

    def public_job(record: JobRecord, request: Request) -> JobResponse:
        return JobResponse(
            **record.model_dump(),
            status_url=str(request.url_for("get_job", job_id=record.job_id)),
            audio_url=str(request.url_for("get_job_audio", job_id=record.job_id)),
        )

    @application.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
        tags=["jobs"],
        name="get_job",
    )
    async def get_job(job_id: str, request: Request) -> JobResponse:
        """Return the current state of a synthesis job."""

        record = await store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return public_job(record, request)

    @application.get(
        "/v1/jobs/{job_id}/audio",
        response_class=FileResponse,
        responses={
            status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
            status.HTTP_409_CONFLICT: {"model": ErrorResponse},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        },
        tags=["jobs"],
        name="get_job_audio",
    )
    async def get_job_audio(job_id: str) -> FileResponse:
        """Return generated WAV audio or a state-specific error."""

        record = await store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if record.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=500,
                detail=record.error or "Synthesis failed.",
            )
        if record.status != JobStatus.SUCCEEDED:
            raise HTTPException(status_code=409, detail="Audio is not ready.")

        audio_path = store.audio_path(job_id)
        if not audio_path.is_file():
            raise HTTPException(
                status_code=500,
                detail="The generated audio file is missing.",
            )
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"{job_id}.wav",
        )

    return application


app = create_app()
