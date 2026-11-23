"""Hardware control backends (Stage 3)."""

from __future__ import annotations

from .gimbal import GimbalController, GimbalUnavailable

__all__ = ["GimbalController", "GimbalUnavailable"]
