"""Mana Persian Piper voice adapter."""

from pathlib import Path

from app.engines.piper_base import PiperEngine


class ManaPiperEngine(PiperEngine):
    """Piper adapter for the Mana Persian medium voice."""

    def __init__(self, *, cache_dir: Path, hf_token: str | None) -> None:
        super().__init__(
            repo_id="MahtaFetrat/Mana-Persian-Piper",
            model_filename="fa_IR-mana-medium.onnx",
            config_filename="fa_IR-mana-medium.onnx.json",
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
