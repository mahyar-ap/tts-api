#!/usr/bin/env python3
"""Import every engine adapter without loading or downloading model weights."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENGINE_MODULES = (
    "app.engines.chatterbox",
    "app.engines.chatterbox_backend",
    "app.engines.chatterbox_worker",
    "app.engines.khadijah_matcha",
    "app.engines.mana_piper",
    "app.engines.manatts_backend",
    "app.engines.manatts_tacotron2",
    "app.engines.manatts_worker",
    "app.engines.piper_base",
    "app.engines.piper_ganji",
    "app.engines.piper_ganji_adabi",
)


def main() -> None:
    """Import all engine modules and report success in English."""

    for module_name in ENGINE_MODULES:
        importlib.import_module(module_name)
        print(f"Imported {module_name}.")
    print("All engine imports completed without loading model weights.")


if __name__ == "__main__":
    main()
