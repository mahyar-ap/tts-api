"""Khadijah Persian-English Matcha TTS adapter for sherpa-onnx."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from app.engines.base import BaseEngine, download_hf_file, float_audio_to_wav_bytes


class KhadijahMatchaEngine(BaseEngine):
    """Run the Khadijah Matcha acoustic model with a universal HiFiGAN."""

    acoustic_repo = "mah92/Khadijah-FA_EN-Matcha-TTS-Model"
    acoustic_filename = "matcha-fa_en-khadijah-22050-5.onnx"
    tokens_filename = "tokens_sherpa_with_fa.txt"
    vocoder_repo = "csukuangfj/sherpa-onnx-hifigan"
    vocoder_filename = "hifigan_v2.onnx"
    espeak_repo = "csukuangfj/matcha-tts-fa_en-khadijah"

    def __init__(
        self,
        *,
        cache_dir: Path,
        hf_token: str | None,
        espeak_data_dir: Path | None,
        num_threads: int,
    ) -> None:
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.espeak_data_dir = espeak_data_dir
        self.num_threads = num_threads
        self._tts: Any | None = None
        self._sherpa: Any | None = None

    def load(self) -> None:
        """Download model assets and construct OfflineTts."""

        if self._tts is not None:
            return
        import sherpa_onnx

        acoustic_model = download_hf_file(
            repo_id=self.acoustic_repo,
            filename=self.acoustic_filename,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        tokens = download_hf_file(
            repo_id=self.acoustic_repo,
            filename=self.tokens_filename,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        vocoder = download_hf_file(
            repo_id=self.vocoder_repo,
            filename=self.vocoder_filename,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        data_dir = self._resolve_espeak_data_dir()

        matcha_config = sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=str(acoustic_model),
            vocoder=str(vocoder),
            tokens=str(tokens),
            data_dir=str(data_dir),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=matcha_config,
            provider="cpu",
            debug=False,
            num_threads=self.num_threads,
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            max_num_sentences=1,
        )
        if hasattr(config, "validate") and not config.validate():
            raise RuntimeError("The sherpa-onnx Matcha configuration is invalid.")
        self._tts = sherpa_onnx.OfflineTts(config)
        self._sherpa = sherpa_onnx

    def synthesize(self, text: str) -> bytes:
        """Generate single-speaker audio at normal speed."""

        if self._tts is None or self._sherpa is None:
            raise RuntimeError("The Khadijah Matcha engine is not loaded.")
        try:
            audio = self._tts.generate(text, sid=0, speed=1.0)
        except TypeError:
            generation = self._sherpa.GenerationConfig()
            generation.sid = 0
            generation.speed = 1.0
            audio = self._tts.generate(text, generation)
        if len(audio.samples) == 0:
            raise RuntimeError("sherpa-onnx returned empty audio.")
        return float_audio_to_wav_bytes(audio.samples, int(audio.sample_rate))

    def unload(self) -> None:
        """Release the sherpa-onnx session."""

        self._tts = None
        self._sherpa = None
        gc.collect()

    def _resolve_espeak_data_dir(self) -> Path:
        if self.espeak_data_dir is not None:
            path = self.espeak_data_dir.resolve()
            if not path.is_dir():
                raise RuntimeError(
                    f"The espeak-ng data directory does not exist: {path}"
                )
            return path

        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=self.espeak_repo,
                allow_patterns=["espeak-ng-data/*"],
                cache_dir=str(self.cache_dir),
                token=self.hf_token,
            )
        )
        path = snapshot / "espeak-ng-data"
        if not path.is_dir():
            raise RuntimeError("The downloaded espeak-ng data directory is missing.")
        return path
