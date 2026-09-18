"""In-process ManaTTS backend used only by the isolated worker environment."""

from __future__ import annotations

import gc
import importlib
import re
import string
import sys
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from app.engines.base import float_audio_to_wav_bytes

_DIACRITICS = re.compile("[\u064b-\u0652\u0654]")
_CHARACTER_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "ئ": "ی",
    }
)
_ALLOWED_CHARACTERS = frozenset(
    "ءابتثجحخدذرزسشصضطظعغفقلمنهویپچژکگآ!(),-.:;?،…؛؟٪#_–@+/ ü" + string.ascii_letters
)


def normalize_persian(text: str) -> str:
    """Remove unsupported marks and normalize Arabic letter variants."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u200c", "")
    normalized = _DIACRITICS.sub("", normalized)
    normalized = normalized.translate(_CHARACTER_TRANSLATION)
    normalized = "".join(
        character if character in _ALLOWED_CHARACTERS else " "
        for character in normalized
    )
    return " ".join(normalized.split())


class ManaTTSBackend:
    """Run the upstream Tacotron2 synthesizer with a CPU HiFiGAN vocoder."""

    def __init__(self, *, repo_dir: Path) -> None:
        self.repo_dir = repo_dir.resolve()
        self.models_dir = self.repo_dir / "saved_models" / "final_models"
        self._encoder: ModuleType | None = None
        self._synthesizer: Any | None = None
        self._vocoder: Any | None = None
        self._torch: ModuleType | None = None

    def load(self) -> None:
        """Load the speaker encoder, synthesizer, and CPU vocoder."""

        if self._synthesizer is not None:
            return
        self._validate_installation()

        repo_path = str(self.repo_dir)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # ParallelWaveGAN 0.6 imports kaiser from its former SciPy location.
        import scipy.signal
        from scipy.signal.windows import kaiser

        if not hasattr(scipy.signal, "kaiser"):
            scipy.signal.kaiser = kaiser

        import torch
        from parallel_wavegan.utils import load_model as load_vocoder

        encoder = importlib.import_module("encoder.inference")
        synthesizer_module = importlib.import_module("synthesizer.inference")
        self._ensure_upstream_module(encoder)
        self._ensure_upstream_module(synthesizer_module)

        encoder.load_model(self.models_dir / "encoder.pt")
        synthesizer = synthesizer_module.Synthesizer(self.models_dir / "synthesizer.pt")
        vocoder = load_vocoder(str(self.models_dir / "vocoder_HiFiGAN.pkl"))
        vocoder.remove_weight_norm()

        self._encoder = encoder
        self._synthesizer = synthesizer
        self._vocoder = vocoder.eval().to("cpu")
        self._torch = torch

    def synthesize(self, text: str) -> bytes:
        """Create a speaker embedding, mel spectrogram, and waveform."""

        if (
            self._encoder is None
            or self._synthesizer is None
            or self._vocoder is None
            or self._torch is None
        ):
            raise RuntimeError("The ManaTTS backend is not loaded.")

        normalized_text = normalize_persian(text)
        if not normalized_text:
            raise ValueError("No supported characters remain after normalization.")

        sample_path = self.models_dir / "sample.wav"
        reference = self._synthesizer.load_preprocess_wav(sample_path)
        encoder_wav = self._encoder.preprocess_wav(reference)
        embedding, _, _ = self._encoder.embed_utterance(
            encoder_wav,
            return_partials=True,
        )
        spectrograms = self._synthesizer.synthesize_spectrograms(
            [normalized_text],
            [embedding],
        )
        spectrogram = np.concatenate(spectrograms, axis=1)
        features = self._torch.from_numpy(spectrogram.T).to("cpu")
        with self._torch.no_grad():
            waveform = self._vocoder.inference(features)
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        return float_audio_to_wav_bytes(
            waveform,
            int(self._synthesizer.sample_rate),
            normalize=True,
        )

    def unload(self) -> None:
        """Release CPU model references."""

        self._encoder = None
        self._synthesizer = None
        self._vocoder = None
        self._torch = None
        gc.collect()

    def _validate_installation(self) -> None:
        if not self.repo_dir.is_dir():
            raise RuntimeError(
                "The ManaTTS repository is missing. Run scripts/setup_manatts.py."
            )
        required = (
            "encoder.pt",
            "synthesizer.pt",
            "vocoder_HiFiGAN.pkl",
            "config.yml",
            "sample.wav",
        )
        missing = [name for name in required if not (self.models_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                "The ManaTTS installation is incomplete. Missing files: "
                + ", ".join(missing)
                + ". Run scripts/setup_manatts.py."
            )

    def _ensure_upstream_module(self, module: ModuleType) -> None:
        module_file = Path(module.__file__ or "").resolve()
        if not module_file.is_relative_to(self.repo_dir):
            raise RuntimeError(
                f"Python module collision detected for {module.__name__}."
            )
