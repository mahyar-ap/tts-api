#!/usr/bin/env python3
"""Validate the isolated ManaTTS package layer and its dependency graph."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

CRITICAL_VERSIONS = {
    "librosa": "0.11.0",
    "numpy": "1.26.4",
    "parallel-wavegan": "0.6.1",
    "scipy": "1.12.0",
    "torch": "2.6.0+cpu",
}


def main() -> None:
    """Reject missing, incompatible, or non-CPU packages in the target layer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    if not package_dir.is_dir():
        raise SystemExit("The ManaTTS package directory does not exist.")

    distributions = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions(path=[str(package_dir)])
        if distribution.metadata["Name"]
    }
    errors: list[str] = []

    for name, expected in CRITICAL_VERSIONS.items():
        distribution = distributions.get(canonicalize_name(name))
        if distribution is None:
            errors.append(f"{name} is not installed")
        elif distribution.version != expected:
            errors.append(
                f"{name} is {distribution.version}, but {expected} is required"
            )

    for distribution in distributions.values():
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate(
                {"extra": ""}
            ):
                continue
            dependency = distributions.get(canonicalize_name(requirement.name))
            if dependency is None:
                errors.append(
                    f"{distribution.metadata['Name']} requires missing "
                    f"{requirement.name}"
                )
            elif (
                requirement.specifier
                and dependency.version not in requirement.specifier
            ):
                errors.append(
                    f"{distribution.metadata['Name']} requires {requirement}, "
                    f"but {dependency.version} is installed"
                )

    if errors:
        print("ManaTTS dependency errors were detected:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}.", file=sys.stderr)
        raise SystemExit(1)

    import torch

    if torch.version.cuda is not None:
        raise SystemExit("The isolated ManaTTS Torch build is not CPU-only.")
    print("The isolated ManaTTS dependency graph is consistent and CPU-only.")


if __name__ == "__main__":
    main()
