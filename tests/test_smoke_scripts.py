"""Tests for smoke scripts that must not download model weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_dependencies import is_expected_transformers_conflict


def test_dependency_checker_allows_only_documented_conflict() -> None:
    expected = (
        "chatterbox-tts 0.1.7 has requirement transformers==5.2.0, "
        "but you have transformers 4.52.0."
    )
    assert is_expected_transformers_conflict(expected) is True
    assert (
        is_expected_transformers_conflict("another-package 1.0 requires torch==9.0")
        is False
    )


def test_engine_import_smoke_script() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/smoke_import_engines.py"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "without loading model weights" in result.stdout
