from __future__ import annotations

"""Reproducibility utilities for random seeds and environment metadata.

The functions in this module reduce uncontrolled variation and record
enough environment information to explain or reproduce a completed run.
"""

import json
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    """
    Seed Python, NumPy, PyTorch and CUDA and request deterministic operations.
    
    Determinism is requested with warnings rather than hard failure because some
    hardware-specific operations may not have a deterministic implementation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Warn instead of failing if the selected hardware lacks a deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str | None:
    """Return the current Git commit hash when execution occurs inside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    """Return an installed package version or None when the package is unavailable."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(device: str) -> dict[str, Any]:
    """
    Collect software, hardware and version information for the current run.
    
    The returned dictionary is saved beside the results to improve auditability.
    """
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": get_git_commit(),
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "matplotlib": package_version("matplotlib"),
            "PyYAML": package_version("PyYAML"),
            "requests": package_version("requests"),
            "torch": package_version("torch"),
        },
    }


def save_run_metadata(
    output_dir: Path,
    raw_config: dict[str, Any],
    device: str,
) -> None:
    """Save the effective YAML configuration and environment metadata with the outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "effective_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)

    with (output_dir / "environment.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(collect_environment(device), handle, indent=2)
