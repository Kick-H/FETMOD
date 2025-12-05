"""Centralized configuration helpers for :mod:`semicon_workflow`.

This module keeps shared constants and lightweight helpers that can be reused
across workflow modules without duplicating magic strings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def resolve_path(path: str | Path) -> Path:
    """Return an absolute, expanded :class:`pathlib.Path` for user supplied locations."""
    return Path(path).expanduser().resolve()


@dataclass(slots=True)
class MPConfig:
    """Materials Project API configuration."""

    api_key: Optional[str] = None

    def get_api_key(self) -> Optional[str]:
        """Return the provided API key or fall back to common environment variables."""
        if self.api_key is not None:
            return self.api_key
        for env_var in ("PMG_MAPI_KEY", "MP_API_KEY"):
            value = os.environ.get(env_var)
            if value:
                return value
        return None
