"""No-op runtime guard for the browser build."""

from __future__ import annotations


def require_compatible_runtime() -> None:
    return None
