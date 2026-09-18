#!/usr/bin/env python3
"""Prepare the upstream ManaTTS Tacotron2 repository and model files."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

UPSTREAM_URL = "https://github.com/MahtaFetrat/Persian-MultiSpeaker-Tacotron2"
MODEL_REPO = "MahtaFetrat/Persian-Tacotron2-on-ManaTTS"


def run(command: list[str]) -> None:
    """Run a setup command and fail on a non-zero exit status."""

    subprocess.run(command, check=True)


def patch_parallel_wavegan() -> None:
    """Patch the SciPy kaiser import in the installed ParallelWaveGAN package."""

    spec = importlib.util.find_spec("parallel_wavegan")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("parallel-wavegan is not installed.")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    pqmf_path = package_dir / "layers" / "pqmf.py"
    source = pqmf_path.read_text("utf-8")
    old = "from scipy.signal import kaiser"
    new = "from scipy.signal.windows import kaiser"
    if old in source:
        pqmf_path.write_text(source.replace(old, new), encoding="utf-8")
        print(f"Patched the ParallelWaveGAN SciPy import in {pqmf_path}.")
    elif new in source:
        print("The ParallelWaveGAN SciPy import is already patched.")
    else:
        raise RuntimeError("The expected kaiser import was not found.")


def patch_librosa_call(repo_dir: Path) -> None:
    """Make the upstream encoder compatible with modern librosa releases."""

    audio_path = repo_dir / "encoder" / "audio.py"
    source = audio_path.read_text("utf-8")
    old = "librosa.resample(wav, source_sr, sampling_rate)"
    new = "librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)"
    if old in source:
        audio_path.write_text(source.replace(old, new), encoding="utf-8")
        print(f"Patched the upstream librosa call in {audio_path}.")
    elif new in source:
        print("The upstream librosa call is already patched.")
    else:
        raise RuntimeError("The expected librosa call was not found.")


def prepare_repository(repo_dir: Path) -> None:
    """Clone the upstream repository when it is not already present."""

    if repo_dir.exists():
        if not (repo_dir / ".git").is_dir():
            raise RuntimeError(
                f"The destination exists but is not the expected repository: {repo_dir}"
            )
        print(f"Using the existing ManaTTS repository at {repo_dir}.")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning the ManaTTS repository into {repo_dir}.")
    run(["git", "clone", "--depth", "1", UPSTREAM_URL, str(repo_dir)])


def copy_hugging_face_file(
    filename: str,
    destination: Path,
    token: str | None,
) -> None:
    """Download one model file and copy it into the final model directory."""

    source = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            token=token,
        )
    )
    shutil.copy2(source, destination)
    print(f"Installed {destination.name}.")


def install_vocoder(models_dir: Path) -> None:
    """Download the VCTK HiFiGAN checkpoint and its matching configuration."""

    patch_parallel_wavegan()
    from parallel_wavegan.utils import download_pretrained_model

    with tempfile.TemporaryDirectory(prefix="manatts-vocoder-") as temporary:
        temporary_dir = Path(temporary)
        print("Downloading the vctk_hifigan.v1 vocoder.")
        download_pretrained_model("vctk_hifigan.v1", str(temporary_dir))
        checkpoint = next(
            temporary_dir.rglob("checkpoint-2500000steps.pkl"),
            None,
        )
        config = next(temporary_dir.rglob("config.yml"), None)
        if checkpoint is None or config is None:
            raise RuntimeError("The downloaded HiFiGAN archive is incomplete.")
        shutil.copy2(checkpoint, models_dir / "vocoder_HiFiGAN.pkl")
        shutil.copy2(config, models_dir / "config.yml")
    print("Installed the HiFiGAN checkpoint and configuration.")


def main() -> None:
    """Prepare all assets used by ManaTTSTacotron2Engine."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("third_party/Persian-MultiSpeaker-Tacotron2"),
        help="Destination for the cloned upstream repository.",
    )
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    prepare_repository(repo_dir)
    patch_librosa_call(repo_dir)

    models_dir = repo_dir / "saved_models" / "final_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo_dir / "saved_models" / "default" / "encoder.pt",
        models_dir / "encoder.pt",
    )
    print("Installed encoder.pt.")

    token = os.getenv("HF_TOKEN")
    copy_hugging_face_file("synthesizer.pt", models_dir / "synthesizer.pt", token)
    copy_hugging_face_file("sample.wav", models_dir / "sample.wav", token)
    install_vocoder(models_dir)
    print("ManaTTS setup completed successfully.")


if __name__ == "__main__":
    main()
