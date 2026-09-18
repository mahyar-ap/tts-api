"""Lazy engine registry with concurrency and unloading policies."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field

from app.config import Settings
from app.engines.base import BaseEngine
from app.engines.chatterbox import ChatterboxEngine
from app.engines.khadijah_matcha import KhadijahMatchaEngine
from app.engines.mana_piper import ManaPiperEngine
from app.engines.manatts_tacotron2 import ManaTTSTacotron2Engine
from app.engines.piper_ganji import PiperGanjiEngine
from app.engines.piper_ganji_adabi import PiperGanjiAdabiEngine
from app.text_chunks import split_sentences_for_stream

logger = logging.getLogger(__name__)
PUBLIC_SYNTHESIS_FAILURE = "The synthesis engine failed to process this request."


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """Factory and memory classification for an exposed engine."""

    model_id: str
    factory: Callable[[], BaseEngine]
    heavy: bool = False


@dataclass(slots=True)
class EngineHandle:
    """Mutable lifecycle state associated with one engine specification."""

    spec: EngineSpec
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    engine: BaseEngine | None = None


class EngineRegistry:
    """Load engines on demand and serialize every call per engine."""

    def __init__(self, specs: list[EngineSpec], unload_policy: str) -> None:
        self._handles = {spec.model_id: EngineHandle(spec) for spec in specs}
        self._unload_policy = unload_policy
        self._lifecycle_lock = asyncio.Lock()

    @property
    def model_ids(self) -> list[str]:
        """Return all configured public model identifiers."""

        return list(self._handles)

    @property
    def loaded_models(self) -> list[str]:
        """Return model identifiers whose engines are currently resident."""

        return [
            model_id
            for model_id, handle in self._handles.items()
            if handle.engine is not None
        ]

    def _require_handle(self, model_id: str) -> EngineHandle:
        handle = self._handles.get(model_id)
        if handle is None:
            raise KeyError(f"Unknown model identifier: {model_id}")
        return handle

    async def ensure_loaded(self, model_id: str) -> list[str]:
        """Load ``model_id``, unloading conflicts per policy. Return loaded ids."""

        handle = self._require_handle(model_id)
        async with self._lifecycle_lock:
            await self._unload_conflicts(handle)
            async with handle.lock:
                await self._load_locked(handle)
        return self.loaded_models

    async def unload_one(self, model_id: str) -> list[str]:
        """Unload a single resident engine. Return remaining loaded ids."""

        handle = self._require_handle(model_id)
        async with self._lifecycle_lock:
            await self._unload(handle)
        return self.loaded_models

    async def synthesize(self, model_id: str, text: str) -> bytes:
        """Ensure an engine is loaded, then run one locked inference call."""

        handle = self._require_handle(model_id)

        # Lock ordering is global lifecycle lock followed by an engine lock.
        # The target lock stays held after the lifecycle lock is released.
        async with self._lifecycle_lock:
            await self._unload_conflicts(handle)
            await handle.lock.acquire()
            try:
                await self._load_locked(handle)
            except Exception:
                handle.lock.release()
                raise

        try:
            assert handle.engine is not None
            return await asyncio.to_thread(handle.engine.synthesize, text)
        finally:
            handle.lock.release()

    async def synthesize_stream(
        self,
        model_id: str,
        text: str,
        *,
        max_chunk_length: int = 300,
    ) -> AsyncIterator[tuple[int, str, bytes]]:
        """Yield ``(index, chunk_text, wav_bytes)`` as audio becomes available."""

        import queue as sync_queue

        handle = self._require_handle(model_id)

        async with self._lifecycle_lock:
            await self._unload_conflicts(handle)
            await handle.lock.acquire()
            try:
                await self._load_locked(handle)
            except Exception:
                handle.lock.release()
                raise

        try:
            assert handle.engine is not None
            engine = handle.engine
            out: sync_queue.Queue[
                tuple[str, int | BaseException | None, str | None, bytes | None]
            ] = sync_queue.Queue(maxsize=2)

            def _produce() -> None:
                try:
                    for index, (chunk_text, wav_bytes) in enumerate(
                        _iter_engine_chunks(engine, text, max_chunk_length)
                    ):
                        out.put(("chunk", index, chunk_text, wav_bytes))
                    out.put(("done", None, None, None))
                except BaseException as exc:  # propagate to the consumer
                    out.put(("error", exc, None, None))

            producer = asyncio.create_task(asyncio.to_thread(_produce))
            try:
                while True:
                    kind, index_or_exc, chunk_text, wav_bytes = await asyncio.to_thread(
                        out.get
                    )
                    if kind == "done":
                        break
                    if kind == "error":
                        assert isinstance(index_or_exc, BaseException)
                        raise index_or_exc
                    assert isinstance(index_or_exc, int)
                    assert isinstance(chunk_text, str)
                    assert isinstance(wav_bytes, bytes)
                    yield index_or_exc, chunk_text, wav_bytes
            finally:
                await producer
        finally:
            handle.lock.release()

    async def unload_all(self) -> None:
        """Unload every resident engine after all inference calls finish."""

        async with self._lifecycle_lock:
            for handle in self._handles.values():
                try:
                    await self._unload(handle)
                except Exception:
                    logger.exception(
                        "Model %s could not be unloaded during shutdown.",
                        handle.spec.model_id,
                    )

    async def _load_locked(self, handle: EngineHandle) -> None:
        if handle.engine is not None:
            return
        model_id = handle.spec.model_id
        engine = handle.spec.factory()
        try:
            logger.info("Loading model %s.", model_id)
            await asyncio.to_thread(engine.load)
            handle.engine = engine
            logger.info("Loaded model %s.", model_id)
        except Exception:
            try:
                await asyncio.to_thread(engine.unload)
            except Exception:
                logger.exception(
                    "Cleanup failed after model %s could not load.", model_id
                )
            raise

    async def _unload_conflicts(self, target: EngineHandle) -> None:
        for handle in self._handles.values():
            if handle is target or handle.engine is None:
                continue
            should_unload = self._unload_policy == "single-model" or (
                self._unload_policy == "single-heavy"
                and target.spec.heavy
                and handle.spec.heavy
            )
            if should_unload:
                await self._unload(handle)

    async def _unload(self, handle: EngineHandle) -> None:
        if handle.engine is None:
            return
        async with handle.lock:
            if handle.engine is None:
                return
            model_id = handle.spec.model_id
            logger.info("Unloading model %s.", model_id)
            engine = handle.engine
            await asyncio.to_thread(engine.unload)
            handle.engine = None
            logger.info("Unloaded model %s.", model_id)


def _iter_engine_chunks(
    engine: BaseEngine,
    text: str,
    max_chunk_length: int,
) -> Iterator[tuple[str, bytes]]:
    """Yield text/audio pairs for streaming, engine-specific when possible."""

    iter_synthesize = getattr(engine, "iter_synthesize", None)
    if callable(iter_synthesize):
        yield from iter_synthesize(text)
        return

    for chunk_text in split_sentences_for_stream(text, max_chunk_length):
        yield chunk_text, engine.synthesize(chunk_text)


def build_engine_specs(settings: Settings) -> list[EngineSpec]:
    """Build the six standard engine factories from application settings."""

    cache_dir = settings.model_cache_dir
    token = settings.hf_token
    return [
        EngineSpec(
            "chatterbox",
            lambda: ChatterboxEngine(
                device=settings.device,
                cache_dir=cache_dir,
                hf_token=token,
                chunk_size=settings.chatterbox_chunk_size,
                startup_timeout_seconds=(
                    settings.chatterbox_startup_timeout_seconds
                ),
                inference_timeout_seconds=(
                    settings.chatterbox_inference_timeout_seconds
                ),
            ),
            heavy=True,
        ),
        EngineSpec(
            "manatts-tacotron2",
            lambda: ManaTTSTacotron2Engine(
                repo_dir=settings.manatts_repo_dir,
                python_executable=settings.manatts_python,
                package_dir=settings.manatts_package_dir,
                startup_timeout_seconds=settings.manatts_startup_timeout_seconds,
                inference_timeout_seconds=settings.manatts_inference_timeout_seconds,
            ),
        ),
        EngineSpec(
            "mana-piper",
            lambda: ManaPiperEngine(cache_dir=cache_dir, hf_token=token),
        ),
        EngineSpec(
            "piper-ganji",
            lambda: PiperGanjiEngine(cache_dir=cache_dir, hf_token=token),
        ),
        EngineSpec(
            "piper-ganji-adabi",
            lambda: PiperGanjiAdabiEngine(cache_dir=cache_dir, hf_token=token),
        ),
        EngineSpec(
            "khadijah-matcha",
            lambda: KhadijahMatchaEngine(
                cache_dir=cache_dir,
                hf_token=token,
                espeak_data_dir=settings.espeak_ng_data_dir,
                num_threads=settings.sherpa_num_threads,
            ),
        ),
    ]


def build_registry(settings: Settings) -> EngineRegistry:
    """Create the default lazy engine registry."""

    return EngineRegistry(
        build_engine_specs(settings),
        unload_policy=settings.model_unload_policy,
    )
