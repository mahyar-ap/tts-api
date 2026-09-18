#!/usr/bin/env python3
"""Load the real Persian Chatterbox model and generate one short WAV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run one authenticated real-model synthesis outside FastAPI."""

    from app.config import get_settings
    from app.engines.chatterbox import ChatterboxEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        default="سلام، این یک آزمایش کوتاه است.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chatterbox-smoke.wav"),
        help="Destination WAV path.",
    )
    args = parser.parse_args()
    settings = get_settings()
    if not settings.hf_token:
        raise SystemExit("HF_TOKEN is required for the real Chatterbox smoke test.")

    engine = ChatterboxEngine(
        device=settings.device,
        cache_dir=settings.model_cache_dir,
        hf_token=settings.hf_token,
    )
    try:
        print("Loading the real Persian Chatterbox model.")
        engine.load()
        print("Generating the Chatterbox smoke-test audio.")
        wav_bytes = engine.synthesize(args.text)
        if not wav_bytes.startswith(b"RIFF"):
            raise RuntimeError("Chatterbox did not return a valid WAV container.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(wav_bytes)
        print(f"Saved the Chatterbox smoke-test WAV to {args.output}.")
    finally:
        engine.unload()


if __name__ == "__main__":
    main()
