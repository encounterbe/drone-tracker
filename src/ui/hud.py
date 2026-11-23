"""Tactical HUD renderer.

Design rules this module follows, in priority order:

1. **Never let the overlay hide the thing it is describing.** Brackets sit
   *outside* the target box, the reticle has a central gap, panels are anchored
   to the frame edges.
2. **Observation and prediction must be visually distinct.** A coasting track -
   one the tracker is extrapolating without any current detection - is drawn
   dashed and labelled ``COAST``. An operator must never mistake the motion
   model's guess for a measurement.
3. **Scale with the frame.** Every size derives from frame height, so the same
   layout works at 720p and 4K without a second set of constants.
4. **Overlay only.** Nothing here feeds back into detection or tracking.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..tracking.track import Track, TrackState
from .theme import Theme, draw_bar, draw_panel, draw_text, alpha_rect

__all__ = ["HUD", "SystemState"]


class SystemState:
    """Top-level system status, derived from the track picture."""

    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    COASTING = "COASTING"
    PAUSED = "PAUSED"


class HUD:
    """Draws the complete operator overlay onto a frame."""

    def __init__(self, cfg, theme: Theme, frame_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.theme = theme
        self.width, self.height = int(frame_size[0]), int(frame_size[1])

        # Single scale factor keeps the layout resolution-independent.
        self.s = self.height / 720.0
        self.font_small = 0.38 * self.s
        self.font_body = 0.44 * self.s
        self.font_title = 0.50 * self.s

        self.show_grid = bool(cfg.get("show_grid", True))
        self.show_trails = bool(cfg.get("show_trails", True))
        self.show_all_tracks = bool(cfg.get("show_all_tracks", True))
        self.show_detections = bool(cfg.get("show_detections", False))
        self.show_pip = bool(cfg.get("show_pip", True))
        self.reticle_scale = float(cfg.get("reticle_scale", 1.0))
        self.scanlines = bool(cfg.get("scanlines", False))

        self._anim = 0.0

    # ==================================================================== API
    def render(
        self,
        view: np.ndarray,
        *,
        tracks: list[Track],
        primary: Track | None,
        viewport,
        focus,
        stats: dict,
        full_frame: np.ndarray | None = None,
        detections: list | None = None,
        paused: bool = False,
    ) -> np.ndarray:
        """Composite the full HUD onto ``view`` (already zoomed). Returns a new image."""
        img = view.copy()
        self._anim += 0.12

        state = self._system_state(tracks, primary, paused)

        if self.show_grid:
            self._draw_grid(img)

        if self.show_detections and detections:
            self._draw_raw_detections(img, detections, viewport)

        if self.show_all_tracks:
            for track in tracks:
                if primary is not None and track.id == primary.id:
                    continue
                if track.state is TrackState.TENTATIVE:
                    self._draw_tentative(img, track, viewport)
                elif track.is_confirmed:
                    self._draw_secondary(img, track, viewport)

        if primary is not None:
            if self.show_trails:
                self._draw_trail(img, primary, viewport)
            self._draw_primary(img, primary, viewport, focus)

        self._draw_reticle(img, state)
        self._draw_status_bar(img, state, stats)
        self._draw_telemetry(img, stats, viewport)
        self._draw_target_panel(img, primary, focus, stats)
        self._draw_footer(img, stats)

        if self.show_pip and viewport.is_zoomed and full_frame is not None:
            self._draw_pip(img, full_frame, viewport, primary)

        if self.scanlines:
            self._draw_scanlines(img)

        return img

    # ============================================================ system state
    @staticmethod
    def _system_state(tracks: list[Track], primary: Track | None, paused: bool) -> str:
        if paused:
            return SystemState.PAUSED
        if primary is None:
            return SystemState.ACQUIRING if tracks else SystemState.SEARCHING
        if primary.is_coasting:
            return SystemState.COASTING
        return SystemState.LOCKED

    def _state_color(self, state: str):
        return {
            SystemState.LOCKED: self.theme.lock,
            SystemState.COASTING: self.theme.warn,
            SystemState.ACQUIRING: self.theme.accent,
            SystemState.SEARCHING: self.theme.secondary,
            SystemState.PAUSED: self.theme.dim,
        }.get(state, self.theme.primary)

    # ================================================================ chrome
    def _draw_grid(self, img: np.ndarray) -> None:
        """Faint thirds grid plus centre axes, for spatial reference."""
        h, w = img.shape[:2]
        color = self.theme.dim
        overlay = img.copy()
        for i in (1, 2):
            x = int(w * i / 3)
            y = int(h * i / 3)
            cv2.line(overlay, (x, 0), (x, h), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)

        # Boresight axes with graduations every 10% of frame width.
        cx, cy = w // 2, h // 2
        tick = int(6 * self.s)
        for frac in np.arange(0.1, 0.51, 0.1):
            dx = int(frac * w / 2)
            for sign in (-1, 1):
                x = cx + sign * dx
                cv2.line(img, (x, cy - tick), (x, cy + tick), self.theme.dim, 1, cv2.LINE_AA)
            dy = int(frac * h / 2)
            for sign in (-1, 1):
                y = cy + sign * dy
                cv2.line(img, (cx - tick, y), (cx + tick, y), self.theme.dim, 1, cv2.LINE_AA)

    def _draw_reticle(self, img: np.ndarray, state: str) -> None:
        """Boresight reticle at frame centre, styled by system state."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        color = self._state_color(state)

        arm = int(26 * self.s * self.reticle_scale)
        gap = int(8 * self.s * self.reticle_scale)

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(
                img,
                (cx + dx * gap, cy + dy * gap),
                (cx + dx * (gap + arm), cy + dy * (gap + arm)),
                color, 1, cv2.LINE_AA,
            )
        cv2.circle(img, (cx, cy), max(1, int(2 * self.s)), color, -1, cv2.LINE_AA)

        if state == SystemState.LOCKED:
            # Static ring: lock achieved, nothing is moving.
            cv2.circle(img, (cx, cy), int(34 * self.s * self.reticle_scale), color, 1, cv2.LINE_AA)
        elif state in (SystemState.SEARCHING, SystemState.ACQUIRING):
            # Rotating dashed ring communicates "actively scanning".
            radius = int(34 * self.s * self.reticle_scale)
            for i in range(8):
                a0 = math.degrees(self._anim * 0.5) + i * 45
                cv2.ellipse(img, (cx, cy), (radius, radius), 0, a0, a0 + 22, color, 1, cv2.LINE_AA)

    def _draw_scanlines(self, img: np.ndarray) -> None:
        overlay = img.copy()
        overlay[::3, :] = (overlay[::3, :] * 0.82).astype(np.uint8)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)

    # ================================================================ targets
    def _draw_primary(self, img: np.ndarray, track: Track, viewport, focus) -> None:
        """Locked target: converging brackets, ID tag, velocity and lead marker."""
        box = viewport.map_box(track.bbox)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        coasting = track.is_coasting
        color = self.theme.warn if coasting else self.theme.lock

        # Brackets converge as the track matures - a readable acquisition cue.
        acquisition = min(1.0, track.hits / 10.0)
        spread = int((1.0 - acquisition) * 26 * self.s)
        x1, y1, x2, y2 = x1 - spread, y1 - spread, x2 + spread, y2 + spread

        bw, bh = x2 - x1, y2 - y1
        corner = max(int(7 * self.s), int(min(bw, bh) * 0.28))
        thickness = max(1, int(2 * self.s))

        for cx_, cy_, dx, dy in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ):
            if coasting:
                self._dashed_line(img, (cx_, cy_), (cx_ + dx * corner, cy_), color, thickness)
                self._dashed_line(img, (cx_, cy_), (cx_, cy_ + dy * corner), color, thickness)
            else:
                cv2.line(img, (cx_, cy_), (cx_ + dx * corner, cy_), color, thickness, cv2.LINE_AA)
                cv2.line(img, (cx_, cy_), (cx_, cy_ + dy * corner), color, thickness, cv2.LINE_AA)

        center = viewport.map_point(track.center)
        ccx, ccy = int(center[0]), int(center[1])

        # Positional uncertainty ellipse: how much the filter trusts itself.
        sigma = viewport.map_length(track.uncertainty)
        if sigma > 2:
            cv2.circle(img, (ccx, ccy), int(min(sigma, 120 * self.s)), self.theme.secondary, 1, cv2.LINE_AA)

        # Velocity vector plus a lead marker 10 frames ahead.
        vel = track.velocity
        speed = float(np.linalg.norm(vel))
        if speed > 0.35:
            lead_src = track.center + vel * 10.0
            lead = viewport.map_point(lead_src)
            cv2.arrowedLine(
                img, (ccx, ccy), (int(lead[0]), int(lead[1])),
                self.theme.accent, max(1, int(1.4 * self.s)), cv2.LINE_AA, tipLength=0.28,
            )
            cv2.drawMarker(
                img, (int(lead[0]), int(lead[1])), self.theme.accent,
                cv2.MARKER_TILTED_CROSS, int(11 * self.s), max(1, int(1.2 * self.s)), cv2.LINE_AA,
            )

        # Tag above the target, flipped below if it would run off the top edge.
        tag = f"TGT-{track.id:03d}"
        sub = "COAST" if coasting else f"{track.confidence * 100:.0f}%"
        ty = y1 - int(10 * self.s)
        if ty < int(22 * self.s):
            ty = y2 + int(20 * self.s)
        draw_text(img, tag, (x1, ty), color, scale=self.font_body, thickness=1)
        draw_text(
            img, sub, (x1, ty + int(14 * self.s)), self.theme.text,
            scale=self.font_small, thickness=1,
        )

        # Corner connector lines toward the boresight, so the eye can always
        # find the target from the centre of the display.
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, h // 2), (ccx, ccy), self.theme.secondary, 1, cv2.LINE_AA)

    def _draw_secondary(self, img: np.ndarray, track: Track, viewport) -> None:
        box = viewport.map_box(track.bbox)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = self.theme.warn if track.is_coasting else self.theme.track
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        draw_text(
            img, f"{track.id:03d}", (x1, y1 - int(5 * self.s)),
            color, scale=self.font_small, thickness=1,
        )

    def _draw_tentative(self, img: np.ndarray, track: Track, viewport) -> None:
        center = viewport.map_point(track.center)
        cv2.drawMarker(
            img, (int(center[0]), int(center[1])), self.theme.tentative,
            cv2.MARKER_SQUARE, int(12 * self.s), 1, cv2.LINE_AA,
        )

    def _draw_raw_detections(self, img: np.ndarray, detections, viewport) -> None:
        for det in detections:
            box = viewport.map_box(det.bbox)
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), self.theme.dim, 1)

    def _draw_trail(self, img: np.ndarray, track: Track, viewport) -> None:
        """Motion history, fading with age."""
        points = list(track.trail)
        if len(points) < 2:
            return
        n = len(points)
        for i in range(1, n):
            p0 = viewport.map_point(points[i - 1])
            p1 = viewport.map_point(points[i])
            fade = i / n
            color = tuple(int(c * (0.25 + 0.75 * fade)) for c in self.theme.track)
            cv2.line(
                img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                color, 1, cv2.LINE_AA,
            )

    def _dashed_line(self, img, p0, p1, color, thickness, dash=6) -> None:
        p0 = np.array(p0, dtype=np.float32)
        p1 = np.array(p1, dtype=np.float32)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1:
            return
        step = max(2.0, dash * self.s)
        n = max(1, int(length / step))
        for i in range(n):
            if i % 2:
                continue
            a = p0 + (p1 - p0) * (i / n)
            b = p0 + (p1 - p0) * (min(i + 1, n) / n)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness, cv2.LINE_AA)

    # ================================================================= panels
    def _draw_status_bar(self, img: np.ndarray, state: str, stats: dict) -> None:
        h, w = img.shape[:2]
        bar_h = int(26 * self.s)
        alpha_rect(img, (0, 0), (w, bar_h), self.theme.panel_bg, self.theme.panel_alpha)
        cv2.line(img, (0, bar_h), (w, bar_h), self.theme.primary, 1, cv2.LINE_AA)

        draw_text(
            img, "UAS DETECTION & TRACKING", (int(12 * self.s), int(18 * self.s)),
            self.theme.primary, scale=self.font_title, thickness=1,
        )

        color = self._state_color(state)
        label = f"[ {state} ]"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, self.font_title, 1)
        draw_text(
            img, label, (w // 2 - tw // 2, int(18 * self.s)),
            color, scale=self.font_title, thickness=1,
        )
        # Blinking marker while coasting: the picture is stale, say so loudly.
        if state == SystemState.COASTING and int(self._anim * 2) % 2 == 0:
            cv2.circle(img, (w // 2 - tw // 2 - int(14 * self.s), int(13 * self.s)),
                       int(4 * self.s), color, -1, cv2.LINE_AA)

        right = (
            f"FRAME {stats.get('frame_index', 0):06d}   "
            f"{stats.get('fps', 0.0):5.1f} FPS"
        )
        (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_DUPLEX, self.font_body, 1)
        draw_text(
            img, right, (w - rw - int(12 * self.s), int(18 * self.s)),
            self.theme.text, scale=self.font_body, thickness=1,
        )

    def _draw_telemetry(self, img: np.ndarray, stats: dict, viewport) -> None:
        pw, ph = int(214 * self.s), int(186 * self.s)
        px, py = int(12 * self.s), int(40 * self.s)
        x, y = draw_panel(img, (px, py), (pw, ph), self.theme, title="SENSOR")

        line_h = int(17 * self.s)
        rows = [
            ("SOURCE", str(stats.get("source_name", "-"))[:16]),
            ("RES", f"{self.width}x{self.height}"),
            ("DETECTOR", str(stats.get("backend", "-")).upper()),
            ("CAM MODE", str(stats.get("camera_mode", "-")).upper()),
            ("EGO-MOT", f"{stats.get('ego_motion', 0.0):.2f} px"),
            ("ZOOM", f"{viewport.zoom:.2f}x" + ("  AUTO" if viewport.auto_follow and viewport.is_zoomed else "")),
            ("CONTACTS", f"{stats.get('detections', 0)}"),
            ("TRACKS", f"{stats.get('confirmed', 0)} / {stats.get('total_tracks', 0)}"),
        ]
        for i, (key, value) in enumerate(rows):
            yy = y + i * line_h
            draw_text(img, key, (x, yy), self.theme.dim, scale=self.font_small, thickness=1)
            draw_text(
                img, value, (x + int(88 * self.s), yy),
                self.theme.text, scale=self.font_small, thickness=1,
            )

    def _draw_target_panel(self, img: np.ndarray, track: Track | None, focus, stats: dict) -> None:
        h, w = img.shape[:2]
        # Height accounts for: title block + 9 data rows + 2 meters + state text.
        pw, ph = int(232 * self.s), int(278 * self.s)
        px, py = w - pw - int(12 * self.s), int(40 * self.s)
        x, y = draw_panel(img, (px, py), (pw, ph), self.theme, title="TARGET")

        line_h = int(17 * self.s)
        if track is None:
            draw_text(img, "NO TARGET", (x, y + line_h), self.theme.dim,
                      scale=self.font_body, thickness=1)
            draw_text(img, "scanning...", (x, y + 2 * line_h), self.theme.dim,
                      scale=self.font_small, thickness=1)
            return

        cx, cy = track.center
        # Offset from boresight in pixels - this is the quantity a gimbal
        # controller consumes, so it is shown in the operator's units directly.
        off_x = cx - self.width / 2.0
        off_y = cy - self.height / 2.0
        fps = max(stats.get("source_fps", 30.0), 1e-6)

        state_color = self.theme.warn if track.is_coasting else self.theme.lock
        rows = [
            ("ID", f"TGT-{track.id:03d}", self.theme.text),
            ("CLASS", track.label[:14], self.theme.text),
            ("STATE", track.state.value, state_color),
            ("POS", f"{cx:6.0f}, {cy:6.0f}", self.theme.text),
            ("OFFSET", f"{off_x:+6.0f}, {off_y:+6.0f}", self.theme.accent),
            ("SIZE", f"{track.size[0]:.0f}x{track.size[1]:.0f} px", self.theme.text),
            ("SPEED", f"{track.speed_px_per_s(fps):6.0f} px/s", self.theme.text),
            ("HEADING", f"{track.heading_deg():5.0f} deg", self.theme.text),
            ("TRK AGE", f"{track.age} f  ({track.hits} hits)", self.theme.text),
        ]
        for i, (key, value, color) in enumerate(rows):
            yy = y + i * line_h
            draw_text(img, key, (x, yy), self.theme.dim, scale=self.font_small, thickness=1)
            draw_text(img, value, (x + int(76 * self.s), yy), color,
                      scale=self.font_small, thickness=1)

        # Confidence + focus meters.
        meter_y = y + len(rows) * line_h + int(6 * self.s)
        bar_w = pw - int(26 * self.s)
        draw_text(img, "CONF", (x, meter_y), self.theme.dim, scale=self.font_small, thickness=1)
        draw_bar(img, (x + int(46 * self.s), meter_y - int(8 * self.s)),
                 (bar_w - int(46 * self.s), int(9 * self.s)),
                 track.confidence, self.theme, color=state_color)

        focus_y = meter_y + int(16 * self.s)
        draw_text(img, "FOCUS", (x, focus_y), self.theme.dim, scale=self.font_small, thickness=1)
        focus_color = self.theme.lock if focus.normalised > 0.7 else self.theme.warn
        draw_bar(img, (x + int(46 * self.s), focus_y - int(8 * self.s)),
                 (bar_w - int(46 * self.s), int(9 * self.s)),
                 focus.normalised, self.theme, color=focus_color)
        draw_text(img, focus.state, (x, focus_y + int(15 * self.s)),
                  focus_color, scale=self.font_small, thickness=1)

    def _draw_footer(self, img: np.ndarray, stats: dict) -> None:
        h, w = img.shape[:2]
        bar_h = int(22 * self.s)
        alpha_rect(img, (0, h - bar_h), (w, h), self.theme.panel_bg, self.theme.panel_alpha)
        cv2.line(img, (0, h - bar_h), (w, h - bar_h), self.theme.secondary, 1, cv2.LINE_AA)

        hint = "SPACE pause  +/- zoom  ARROWS pan  T cycle target  F focus  H hud  M mask  S snapshot  Q quit"
        draw_text(img, hint, (int(12 * self.s), h - int(7 * self.s)),
                  self.theme.dim, scale=self.font_small, thickness=1)

        timings = stats.get("timings", {})
        if timings:
            text = "  ".join(f"{k}:{v:.1f}ms" for k, v in timings.items())
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, self.font_small, 1)
            draw_text(img, text, (w - tw - int(12 * self.s), h - int(7 * self.s)),
                      self.theme.dim, scale=self.font_small, thickness=1)

    def _draw_pip(self, img: np.ndarray, full_frame: np.ndarray, viewport, primary) -> None:
        """Picture-in-picture overview showing what the zoom window is cropping.

        Without this, an operator at 8x has no idea where in the sensor's field
        of view they are looking - which is how you lose situational awareness
        and miss a second contact entering the frame.
        """
        h, w = img.shape[:2]
        pip_w = int(w * 0.18)
        pip_h = int(pip_w * self.height / max(self.width, 1))
        margin = int(12 * self.s)
        px, py = w - pip_w - margin, h - pip_h - int(30 * self.s)

        thumb = cv2.resize(full_frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
        thumb = (thumb.astype(np.float32) * 0.82).astype(np.uint8)
        img[py : py + pip_h, px : px + pip_w] = thumb
        cv2.rectangle(img, (px, py), (px + pip_w, py + pip_h), self.theme.primary, 1, cv2.LINE_AA)

        # The zoom window, mapped into thumbnail coordinates.
        rx, ry, rw, rh = viewport.roi
        sx = pip_w / float(self.width)
        sy = pip_h / float(self.height)
        cv2.rectangle(
            img,
            (px + int(rx * sx), py + int(ry * sy)),
            (px + int((rx + rw) * sx), py + int((ry + rh) * sy)),
            self.theme.accent, 1, cv2.LINE_AA,
        )

        if primary is not None:
            tc = primary.center
            cv2.drawMarker(
                img, (px + int(tc[0] * sx), py + int(tc[1] * sy)),
                self.theme.lock, cv2.MARKER_CROSS, int(8 * self.s), 1, cv2.LINE_AA,
            )

        draw_text(img, "FOV", (px + 4, py + int(12 * self.s)),
                  self.theme.primary, scale=self.font_small, thickness=1)

    # ----------------------------------------------------------------- extras
    def draw_mask_inset(self, img: np.ndarray, mask: np.ndarray) -> None:
        """Debug: show the detector's working mask as a small inset."""
        if mask is None or mask.size == 0:
            return
        h, w = img.shape[:2]
        iw = int(w * 0.18)
        ih = int(iw * mask.shape[0] / max(mask.shape[1], 1))
        small = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
        colored = cv2.applyColorMap(small, cv2.COLORMAP_BONE)
        x, y = int(12 * self.s), h - ih - int(34 * self.s)
        img[y : y + ih, x : x + iw] = colored
        cv2.rectangle(img, (x, y), (x + iw, y + ih), self.theme.secondary, 1, cv2.LINE_AA)
        draw_text(img, "MOTION MASK", (x + 4, y + int(12 * self.s)),
                  self.theme.primary, scale=self.font_small, thickness=1)
