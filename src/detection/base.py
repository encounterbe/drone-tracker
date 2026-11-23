"""Detection data contract and detector interface.

Everything downstream of detection - fusion, tracking, HUD, export - speaks
``Detection``. Adding a new detector means implementing one method and emitting
this type; no other module changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Detection", "Detector", "DetectorUnavailable"]


class DetectorUnavailable(RuntimeError):
    """Raised when a detector's dependencies or weights are missing.

    The pipeline catches this at construction time and degrades to a detector
    that is available, rather than dying at frame 1.
    """


@dataclass
class Detection:
    """A single object observation in one frame.

    Attributes:
        bbox: ``[x1, y1, x2, y2]`` float32, image pixels.
        score: Confidence in ``[0, 1]``.
        label: Human-readable class name.
        source: Which detector produced it (``motion``/``cnn``/``fusion``).
        meta: Free-form diagnostics (blob area, contrast, contributing sources).
    """

    bbox: np.ndarray
    score: float
    label: str = "CONTACT"
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        self.score = float(np.clip(self.score, 0.0, 1.0))

    # ------------------------------------------------------------- geometry
    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def aspect(self) -> float:
        h = self.height
        return self.width / h if h > 1e-6 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used by the track exporter."""
        return {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "score": round(self.score, 4),
            "label": self.label,
            "source": self.source,
            "meta": {
                k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.meta.items()
            },
        }


class Detector(ABC):
    """Base class for all detection backends."""

    name: str = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        """Return detections for one BGR frame.

        Implementations must be pure with respect to the caller: they may keep
        internal temporal state (background models, previous frames), but must
        not modify ``frame`` in place.
        """

    def reset(self) -> None:
        """Drop temporal state. Called on seek/loop so models do not smear."""

    @property
    def debug_mask(self) -> np.ndarray | None:
        """Optional single-channel visualisation of the detector's working mask."""
        return None
