"""Ganji-Adabi Persian Piper voice adapter."""

from pathlib import Path

from app.engines.piper_base import PiperEngine


class PiperGanjiAdabiEngine(PiperEngine):
    """Piper adapter for the literary Ganji-Adabi medium voice."""

    def __init__(self, *, cache_dir: Path, hf_token: str | None) -> None:
        model = "fa/fa_IR/ganji_adabi/medium/fa_IR-ganji_adabi-medium.onnx"
        super().__init__(
            repo_id="rhasspy/piper-voices",
            model_filename=model,
            config_filename=f"{model}.json",
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
