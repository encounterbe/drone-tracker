"""Digital zoom viewport with source<->screen coordinate mapping.

The subtle part of any zoom implementation is not the cropping, it is keeping
overlay geometry consistent with it. Detection and tracking always work in
*source* pixel coordinates; the HUD draws in *viewport* coordinates. Every
overlay therefore has to be mapped through the same transform the image went
through, or the reticle drifts off the target exactly when zoomed in - which is
when precision matters most.

Clamping is the trap: at high zoom near a frame edge the requested window would
extend outside the image, so the actual window is shifted back inside. The
mapping is consequently derived from the *clamped* ROI that was really used,
never from the requested centre.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["Viewport"]


class Viewport:
    """Crop-and-scale viewport providing digital zoom and target following."""

    def __init__(self, cfg, frame_size: tuple[int, int]) -> None:
        """Args:
        cfg: The ``zoom`` config node.
        frame_size: ``(width, height)`` of the source frames.
        """
        self.enabled = bool(cfg.get("enabled", True))
        self.min_zoom = float(cfg.get("min", 1.0))
        self.max_zoom = float(cfg.get("max", 8.0))
        self.step = float(cfg.get("step", 1.25))
        self.auto_follow = bool(cfg.get("auto_follow", True))
        self.smoothing = float(cfg.get("follow_smoothing", 0.15))
        self.auto_zoom = bool(cfg.get("auto_zoom", False))
        self.auto_zoom_fill = float(cfg.get("auto_zoom_target_fill", 0.22))

        self.width, self.height = int(frame_size[0]), int(frame_size[1])
        self.zoom = float(np.clip(cfg.get("initial", 1.0), self.min_zoom, self.max_zoom))

        # Requested centre in source coordinates (may be outside the legal range;
        # the ROI computation clamps it).
        self._center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
        self._roi = (0, 0, self.width, self.height)  # x, y, w, h actually used
        self._manual_offset = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------ state
    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def roi(self) -> tuple[int, int, int, int]:
        """The source rectangle currently displayed: ``(x, y, w, h)``."""
        return self._roi

    @property
    def is_zoomed(self) -> bool:
        return self.zoom > 1.001

    # --------------------------------------------------------------- controls
    def zoom_in(self) -> None:
        self.zoom = float(min(self.max_zoom, self.zoom * self.step))

    def zoom_out(self) -> None:
        self.zoom = float(max(self.min_zoom, self.zoom / self.step))
        if not self.is_zoomed:
            self._manual_offset[:] = 0.0

    def set_zoom(self, value: float) -> None:
        self.zoom = float(np.clip(value, self.min_zoom, self.max_zoom))

    def reset(self) -> None:
        self.zoom = self.min_zoom
        self._center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
        self._manual_offset[:] = 0.0

    def pan(self, dx: float, dy: float) -> None:
        """Nudge the zoom window manually, in source pixels."""
        if not self.is_zoomed:
            return
        self._manual_offset += np.array([dx, dy], dtype=np.float32)

    # ----------------------------------------------------------------- update
    def update(self, target_center: np.ndarray | None, target_size: np.ndarray | None = None) -> None:
        """Advance follow/auto-zoom state for this frame.

        Args:
            target_center: Locked target centre in source coords, or ``None``.
            target_size: Locked target ``(w, h)``, used only by auto-zoom.
        """
        if not self.enabled:
            return

        if self.auto_zoom and target_size is not None and self.is_zoomed:
            # Choose a zoom that makes the target occupy `auto_zoom_fill` of the
            # shorter viewport edge, so it stays comfortably framed as range changes.
            target_extent = float(max(target_size[0], target_size[1]))
            if target_extent > 1.0:
                desired_view = target_extent / max(self.auto_zoom_fill, 1e-3)
                desired_zoom = float(min(self.width, self.height)) / max(desired_view, 1.0)
                desired_zoom = float(np.clip(desired_zoom, self.min_zoom, self.max_zoom))
                # Ease toward it; snapping zoom every frame is unwatchable.
                self.zoom += (desired_zoom - self.zoom) * 0.06

        if self.auto_follow and target_center is not None and self.is_zoomed:
            desired = np.asarray(target_center, dtype=np.float32) + self._manual_offset
            self._center += (desired - self._center) * float(np.clip(self.smoothing, 0.01, 1.0))
        elif not self.is_zoomed:
            self._center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
        else:
            self._center = (
                np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)
                + self._manual_offset
            )

    # ---------------------------------------------------------------- render
    def _compute_roi(self) -> tuple[int, int, int, int]:
        view_w = max(8.0, self.width / self.zoom)
        view_h = max(8.0, self.height / self.zoom)

        cx = float(np.clip(self._center[0], view_w / 2.0, self.width - view_w / 2.0))
        cy = float(np.clip(self._center[1], view_h / 2.0, self.height - view_h / 2.0))

        x = int(round(cx - view_w / 2.0))
        y = int(round(cy - view_h / 2.0))
        w = int(round(view_w))
        h = int(round(view_h))

        x = max(0, min(x, self.width - w))
        y = max(0, min(y, self.height - h))
        return x, y, max(1, w), max(1, h)

    def render(self, frame: np.ndarray, interpolation: int | None = None) -> np.ndarray:
        """Return the zoomed view at full output resolution.

        Upscaling uses Lanczos: at 4-8x on a small target, bilinear turns the
        airframe into mush, and the operator is looking at this view precisely
        to make an identification call.
        """
        if not self.enabled or not self.is_zoomed:
            self._roi = (0, 0, self.width, self.height)
            return frame

        x, y, w, h = self._compute_roi()
        self._roi = (x, y, w, h)
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            self._roi = (0, 0, self.width, self.height)
            return frame

        interp = interpolation if interpolation is not None else cv2.INTER_LANCZOS4
        return cv2.resize(crop, (self.width, self.height), interpolation=interp)

    # --------------------------------------------------------------- mapping
    @property
    def scale(self) -> tuple[float, float]:
        """Source-to-viewport scale factors ``(sx, sy)``."""
        _, _, w, h = self._roi
        return self.width / float(max(w, 1)), self.height / float(max(h, 1))

    def map_point(self, point) -> np.ndarray:
        """Map a source-space point into viewport space."""
        x0, y0, _, _ = self._roi
        sx, sy = self.scale
        p = np.asarray(point, dtype=np.float32).reshape(2)
        return np.array([(p[0] - x0) * sx, (p[1] - y0) * sy], dtype=np.float32)

    def map_box(self, box) -> np.ndarray:
        """Map a source-space ``[x1,y1,x2,y2]`` box into viewport space."""
        x0, y0, _, _ = self._roi
        sx, sy = self.scale
        b = np.asarray(box, dtype=np.float32).reshape(4)
        return np.array(
            [(b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy],
            dtype=np.float32,
        )

    def map_length(self, length: float) -> float:
        """Map a scalar length (e.g. a radius) using the mean scale factor."""
        sx, sy = self.scale
        return float(length) * 0.5 * (sx + sy)

    def is_visible(self, point) -> bool:
        """Whether a source-space point currently falls inside the view."""
        x0, y0, w, h = self._roi
        p = np.asarray(point, dtype=np.float32).reshape(2)
        return bool(x0 <= p[0] <= x0 + w and y0 <= p[1] <= y0 + h)
