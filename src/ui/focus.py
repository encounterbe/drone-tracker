"""Focus measurement and focus assist.

An honest note on what this is. With a fixed-focus USB camera or a recorded MP4
there is no focus motor to drive, so "autofocus" in the mechanical sense is not
something software can provide. What *is* both real and useful is the pair of
functions a professional camera system exposes to its operator:

**Focus measurement** - quantify how sharp the target currently is, so the
operator (or, later, a lens control loop) can tell in-focus from soft. Sharpness
is the variance of the Laplacian over the target ROI: a well-focused edge
produces large second derivatives, a blurred one does not.

**Focus assist / peaking** - enhance the displayed image so residual softness is
easier to judge by eye. This is a *display* transform only; detection always
runs on the untouched frame, so the assist can never invent or destroy contacts.

If a motorised lens is added later, ``FocusAnalyzer.trend`` already provides the
gradient signal a hill-climbing autofocus loop needs: drive the motor in the
direction that increases sharpness, reverse when it starts to fall.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

__all__ = ["FocusAnalyzer", "FocusState"]


class FocusState:
    """Qualitative focus verdict shown on the HUD."""

    SHARP = "IN FOCUS"
    ACCEPTABLE = "ACCEPTABLE"
    SOFT = "SOFT"
    HUNTING = "HUNTING"
    NO_TARGET = "NO TARGET"


class FocusAnalyzer:
    """Measures target sharpness over time and applies optional focus assist."""

    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.get("enabled", True))
        self.reference = max(1.0, float(cfg.get("reference_sharpness", 420.0)))
        self.window = int(cfg.get("metric_window", 30))
        self.assist_enabled = bool(cfg.get("assist", True))
        self.assist_amount = float(cfg.get("assist_amount", 0.6))
        self.assist_radius = int(cfg.get("assist_radius", 3))

        self._history: deque[float] = deque(maxlen=self.window)
        self.last_value: float = 0.0
        self.state: str = FocusState.NO_TARGET

    # ---------------------------------------------------------- measurement
    @staticmethod
    def measure(roi: np.ndarray) -> float:
        """Variance of the Laplacian over a ROI. Higher is sharper."""
        if roi is None or roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def update(self, frame: np.ndarray, bbox: np.ndarray | None) -> float:
        """Measure sharpness inside ``bbox`` and update the rolling history.

        The ROI is padded outward: a tight box around a small target contains
        few edge pixels, and the sharpest edge of all is the target's silhouette
        against the sky, which sits exactly on the box boundary.
        """
        if not self.enabled or bbox is None:
            self.state = FocusState.NO_TARGET
            self.last_value = 0.0
            return 0.0

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        pad = max(6.0, 0.35 * max(x2 - x1, y2 - y1))
        ix1 = int(max(0, x1 - pad))
        iy1 = int(max(0, y1 - pad))
        ix2 = int(min(w, x2 + pad))
        iy2 = int(min(h, y2 + pad))
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            self.state = FocusState.NO_TARGET
            self.last_value = 0.0
            return 0.0

        value = self.measure(frame[iy1:iy2, ix1:ix2])
        self._history.append(value)
        self.last_value = value
        self.state = self._classify(value)
        return value

    def _classify(self, value: float) -> str:
        normalised = value / self.reference
        # A rapidly changing metric means focus (or range) is still settling.
        if len(self._history) >= 6 and abs(self.trend) > 0.28:
            return FocusState.HUNTING
        if normalised >= 0.75:
            return FocusState.SHARP
        if normalised >= 0.35:
            return FocusState.ACCEPTABLE
        return FocusState.SOFT

    # --------------------------------------------------------------- signals
    @property
    def normalised(self) -> float:
        """Sharpness scaled to [0, 1] against the configured reference."""
        return float(np.clip(self.last_value / self.reference, 0.0, 1.0))

    @property
    def trend(self) -> float:
        """Relative change across the history window.

        Positive means sharpness is improving. This is the error signal a
        closed-loop lens controller would consume.
        """
        if len(self._history) < 6:
            return 0.0
        values = np.asarray(self._history, dtype=np.float64)
        half = len(values) // 2
        older = float(values[:half].mean())
        newer = float(values[half:].mean())
        if older <= 1e-6:
            return 0.0
        return float((newer - older) / older)

    @property
    def peak(self) -> float:
        return float(max(self._history)) if self._history else 0.0

    def reset(self) -> None:
        self._history.clear()
        self.last_value = 0.0
        self.state = FocusState.NO_TARGET

    # ----------------------------------------------------------------- assist
    def apply_assist(self, image: np.ndarray) -> np.ndarray:
        """Unsharp-mask an image for display. Returns a new array.

        Display-only: never feed the result back into detection, or the added
        edge energy will change what the detector sees frame to frame.
        """
        if not (self.enabled and self.assist_enabled) or self.assist_amount <= 0:
            return image
        radius = max(1, self.assist_radius)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
        return cv2.addWeighted(image, 1.0 + self.assist_amount, blurred, -self.assist_amount, 0)

    @staticmethod
    def peaking_overlay(image: np.ndarray, color: tuple[int, int, int], threshold: int = 55) -> np.ndarray:
        """Classic focus peaking: tint the highest-gradient pixels.

        Gives an at-a-glance answer to "is the target itself sharp, or only the
        clutter behind it" - the two look identical on a sharpness number alone.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        magnitude = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5,
                                                        cv2.convertScaleAbs(gy), 0.5, 0))
        mask = magnitude > threshold
        out = image.copy()
        out[mask] = color
        return out
