"""Configuration loading with paths anchored to the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_config() -> dict[str, Any]:
    return load_yaml("configs/model.yaml")


def policy_config() -> dict[str, Any]:
    return load_yaml("configs/policy.yaml")

