"""Detection backends and the factory that builds them from config."""

from __future__ import annotations

import logging

from .base import Detection, Detector, DetectorUnavailable
from .fusion import FusionDetector
from .motion import MotionDetector

__all__ = [
    "Detection",
    "Detector",
    "DetectorUnavailable",
    "MotionDetector",
    "FusionDetector",
    "build_detector",
]

log = logging.getLogger(__name__)


def build_detector(cfg) -> Detector:
    """Construct the detector stack named by ``cfg.detection.backend``.

    Degrades gracefully: if a CNN backend is requested but unavailable, the
    reason is logged and motion detection is used instead, so a missing optional
    dependency never takes the system down.
    """
    backend = str(cfg.detection.get("backend", "motion")).lower()

    if backend == "motion":
        return MotionDetector(cfg.detection.motion)

    if backend in ("cnn", "fusion"):
        from .cnn import CNNDetector  # lazy: keeps ultralytics optional

        try:
            cnn = CNNDetector(cfg.detection.cnn)
        except DetectorUnavailable as exc:
            log.warning("CNN backend unavailable (%s) - falling back to motion detection", exc)
            return MotionDetector(cfg.detection.motion)

        if backend == "cnn":
            return cnn
        return FusionDetector([MotionDetector(cfg.detection.motion), cnn], cfg.detection.fusion)

    raise ValueError(
        f"Unknown detection backend {backend!r}; expected one of: motion, cnn, fusion"
    )
