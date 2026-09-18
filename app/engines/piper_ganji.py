"""Ganji Persian Piper voice adapter."""

from pathlib import Path

from app.engines.piper_base import PiperEngine


class PiperGanjiEngine(PiperEngine):
    """Piper adapter for the conversational Ganji medium voice."""

    def __init__(self, *, cache_dir: Path, hf_token: str | None) -> None:
        model = "fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx"
        super().__init__(
            repo_id="rhasspy/piper-voices",
            model_filename=model,
            config_filename=f"{model}.json",
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
