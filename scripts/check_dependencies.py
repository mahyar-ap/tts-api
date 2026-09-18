#!/usr/bin/env python3
"""Validate the tested stack while allowing one documented metadata override."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys

EXPECTED_VERSIONS = {
    "chatterbox-tts": "0.1.7",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.52.0",
}


def is_expected_transformers_conflict(line: str) -> bool:
    """Return whether a pip-check line is the deliberate Chatterbox override."""

    normalized = line.lower().replace("_", "-")
    return (
        "chatterbox-tts 0.1.7" in normalized
        and "transformers==5.2.0" in normalized
        and "transformers 4.52.0" in normalized
    )


def main() -> None:
    """Check critical versions and reject every unexpected dependency error."""

    mismatches: list[str] = []
    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{distribution} is not installed")
            continue
        public_version = actual.split("+", maxsplit=1)[0]
        if public_version != expected:
            mismatches.append(
                f"{distribution} is {actual}, but the tested version is {expected}"
            )

    if mismatches:
        for mismatch in mismatches:
            print(f"Dependency error: {mismatch}.", file=sys.stderr)
        raise SystemExit(1)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    pip_check_lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    unexpected = [
        line for line in pip_check_lines if not is_expected_transformers_conflict(line)
    ]
    if unexpected:
        print("Unexpected dependency conflicts were detected:", file=sys.stderr)
        for line in unexpected:
            print(f"- {line}", file=sys.stderr)
        raise SystemExit(1)

    print("Critical dependency versions match the tested Persian stack.")
    if pip_check_lines:
        print(
            "The known Chatterbox Transformers metadata conflict is intentionally "
            "overridden."
        )


if __name__ == "__main__":
    main()
