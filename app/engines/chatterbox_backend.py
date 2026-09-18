"""In-process Chatterbox backend imported only by the isolated worker."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from app.engines.base import float_audio_to_wav_bytes


class _NoOpWatermarker:
    """Stand-in when resemble-perth fails to export PerthImplicitWatermarker."""

    def apply_watermark(self, wav, sample_rate=None):  # noqa: ANN001
        return wav


def _patch_perth_watermarker() -> None:
    """Python 3.12 often leaves PerthImplicitWatermarker as None; Chatterbox then crashes."""

    try:
        import perth
    except ImportError:
        return
    if perth.PerthImplicitWatermarker is not None:
        return
    perth.PerthImplicitWatermarker = _NoOpWatermarker


class ChatterboxBackend:
    """Load the Chatterbox base model with the Persian T3 checkpoint."""

    repo_id = "Thomcles/Chatterbox-TTS-Persian-Farsi"
    checkpoint_filename = "t3_fa.safetensors"

    def __init__(
        self,
        *,
        device: str,
        cache_dir: Path,
        hf_token: str,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self._model: Any | None = None

    def load(self) -> None:
        """Load base weights and replace the T3 component with Persian weights."""

        if self._model is not None:
            return

        _patch_perth_watermarker()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        if getattr(model, "watermarker", None) is None:
            model.watermarker = _NoOpWatermarker()
        checkpoint = Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.checkpoint_filename,
                cache_dir=str(self.cache_dir),
                token=self.hf_token,
            )
        )
        state_dict = load_file(str(checkpoint), device="cpu")
        model.t3.load_state_dict(state_dict, strict=True)
        model.t3.to(self.device).eval()
        self._model = model
        self._warmup()

    def _warmup(self) -> None:
        """First Chatterbox generate() compiles CUDA kernels (~20-30s on a 1080 Ti)."""

        import time

        started = time.perf_counter()
        self.synthesize("سلام.")
        elapsed = int(round((time.perf_counter() - started) * 1000))
        print(f"Chatterbox warmup generate={elapsed}ms", flush=True)

    def synthesize(self, text: str) -> bytes:
        """Generate one bounded Persian text chunk."""

        if self._model is None:
            raise RuntimeError("The Chatterbox backend is not loaded.")
        import torch

        with torch.inference_mode():
            waveform = self._model.generate(
                text=text,
                language_id=None,
                audio_prompt_path=None,
                temperature=0.7,
                cfg_weight=0.5,
                top_p=0.5,
                exaggeration=0.6,
            )
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        return float_audio_to_wav_bytes(waveform, int(self._model.sr))

    def unload(self) -> None:
        """Release Chatterbox and return unused CUDA allocations."""

        model = self._model
        self._model = None
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
