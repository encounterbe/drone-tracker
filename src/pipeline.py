"""Frame-loop orchestration.

This module owns the order of operations and nothing else; every algorithm lives
in ``detection/``, ``tracking/`` or ``ui/``. Keeping the loop this thin is what
makes the planned Stage-3 gimbal integration a matter of adding one subscriber
to ``primary`` rather than an invasive change.

Per frame::

    capture -> detect (every Nth) -> track -> select primary
            -> focus measure -> viewport update -> HUD render
            -> display / record / export

Note the split between *source* and *display* images: detection and tracking
always consume the untouched frame, while zoom and focus assist only ever affect
what is shown. An operator aid must never be able to change what the system
believes it sees.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .detection import Detection, build_detector
from .tracking import MultiObjectTracker, Track
from .ui import HUD, FocusAnalyzer, Viewport, get_theme
from .utils.perf import PerfMeter
from .utils.video import VideoRecorder, VideoSource

log = logging.getLogger(__name__)

__all__ = ["Pipeline", "PipelineResult"]

WINDOW_NAME = "UAS Detection & Tracking"


@dataclass
class PipelineResult:
    """Summary returned when a run finishes."""

    frames_processed: int = 0
    total_detections: int = 0
    tracks_created: int = 0
    frames_with_lock: int = 0
    mean_fps: float = 0.0
    elapsed: float = 0.0
    recording_path: str | None = None
    export_path: str | None = None
    per_frame: list[dict[str, Any]] = field(default_factory=list)

    @property
    def lock_rate(self) -> float:
        """Fraction of processed frames that had a locked target."""
        return self.frames_with_lock / self.frames_processed if self.frames_processed else 0.0

    def summary(self) -> str:
        return (
            f"{self.frames_processed} frames | {self.mean_fps:.1f} FPS avg | "
            f"{self.total_detections} detections | {self.tracks_created} tracks | "
            f"lock on {self.lock_rate * 100:.1f}% of frames"
        )


class Pipeline:
    """Runs detection, tracking and the operator display over a video source."""

    def __init__(
        self,
        source: str | int,
        cfg,
        *,
        display: bool = True,
        record_path: str | None = None,
        export_path: str | None = None,
        snapshot_dir: str = "output/snapshots",
        start_frame: int = 0,
        max_frames: int = 0,
        realtime: bool = False,
        on_frame: Callable[[dict], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.display = bool(display)
        self.record_path = record_path
        self.export_path = export_path
        self.snapshot_dir = snapshot_dir
        self.realtime = bool(realtime)
        self.on_frame = on_frame

        self.source = VideoSource(
            source,
            process_width=int(cfg.video.get("process_width", 0)),
            loop=bool(cfg.video.get("loop", False)),
            start_frame=start_frame,
            max_frames=max_frames,
        )
        info = self.source.info
        frame_size = (info.width, info.height)
        log.info(
            "Source %s | %dx%d @ %.2f fps | %s frames",
            info.path, info.width, info.height, info.fps,
            info.frame_count if info.frame_count else "unknown",
        )

        self.detector = build_detector(cfg)
        self.detect_interval = max(1, int(cfg.detection.get("interval", 1)))
        self.tracker = MultiObjectTracker(cfg.tracking, trail_length=int(cfg.ui.get("trail_length", 48)))
        self.viewport = Viewport(cfg.zoom, frame_size)
        self.focus = FocusAnalyzer(cfg.focus)
        self.theme = get_theme(cfg.ui.get("theme", "tactical_amber"))
        self.hud = HUD(cfg.ui, self.theme, frame_size)
        self.perf = PerfMeter(window=60)

        self.show_hud = bool(cfg.ui.get("show_hud", True))
        self.show_mask = bool(cfg.ui.get("show_mask", False))

        self.recorder = (
            VideoRecorder(record_path, info.fps, str(cfg.export.get("fourcc", "mp4v")))
            if record_path else None
        )
        self._export_fh = None
        if export_path:
            parent = os.path.dirname(os.path.abspath(export_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._export_fh = open(export_path, "w", encoding="utf-8")
        self._export_tentative = bool(cfg.export.get("include_tentative", False))

        self._paused = False
        self._step_once = False
        self._quit = False
        self._latest_detections: list[Detection] = []
        self._seen_track_ids: set[int] = set()

    # ==================================================================== run
    def run(self) -> PipelineResult:
        result = PipelineResult(recording_path=self.record_path, export_path=self.export_path)
        info = self.source.info

        if self.display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, info.width, info.height)

        frame_budget = 1.0 / info.fps if (self.realtime and info.fps > 0) else 0.0

        try:
            for frame_index, frame in self.source:
                loop_start = time.perf_counter()

                record = self._process_frame(frame_index, frame, result)
                result.per_frame.append(record)

                if self.on_frame is not None:
                    self.on_frame(record)

                self.perf.tick()
                result.frames_processed += 1

                if self.display:
                    if not self._handle_keys():
                        break
                    # While paused, keep servicing the UI so the window stays live.
                    while self._paused and not self._step_once and not self._quit:
                        if not self._handle_keys():
                            break
                    if self._step_once:
                        self._step_once = False

                if self._quit:
                    break

                if frame_budget:
                    remaining = frame_budget - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self._cleanup()

        result.mean_fps = self.perf.mean_fps
        result.elapsed = self.perf.elapsed
        result.tracks_created = len(self._seen_track_ids)
        return result

    # ========================================================== frame handling
    def _process_frame(self, frame_index: int, frame: np.ndarray, result: PipelineResult) -> dict:
        # --- detection (optionally strided) ---
        with self.perf.stage("detect"):
            detections_valid = frame_index % self.detect_interval == 0
            if detections_valid:
                detections = self.detector.detect(frame, frame_index)
                self._latest_detections = detections
            else:
                # Between detection frames the tracker coasts on its motion
                # model; feeding stale boxes would corrupt the filter.
                detections = []

        # --- tracking ---
        with self.perf.stage("track"):
            # The empty list above must not read as "the detector found
            # nothing" - it means the detector never ran. See Track.predict.
            tracks = self.tracker.update(
                detections, frame_index, frame.shape[:2], detections_valid=detections_valid
            )
            primary = self.tracker.primary
            for t in tracks:
                self._seen_track_ids.add(t.id)

        # --- focus measurement on the untouched frame ---
        with self.perf.stage("focus"):
            sharpness = self.focus.update(frame, primary.bbox if primary is not None else None)
            if primary is not None:
                primary.sharpness_history.append(sharpness)

        # --- display construction ---
        with self.perf.stage("render"):
            self.viewport.update(
                primary.center if primary is not None else None,
                primary.size if primary is not None else None,
            )
            view = self.viewport.render(frame)
            if self.viewport.is_zoomed:
                view = self.focus.apply_assist(view)

            if self.show_hud:
                stats = self._build_stats(frame_index, detections, tracks)
                view = self.hud.render(
                    view,
                    tracks=tracks,
                    primary=primary,
                    viewport=self.viewport,
                    focus=self.focus,
                    stats=stats,
                    full_frame=frame,
                    detections=self._latest_detections,
                    paused=self._paused,
                )
                if self.show_mask:
                    self.hud.draw_mask_inset(view, self.detector.debug_mask)

            self._display_frame = view

        if self.display:
            cv2.imshow(WINDOW_NAME, view)
        if self.recorder is not None:
            self.recorder.write(view)

        result.total_detections += len(detections)
        if primary is not None and not primary.is_coasting:
            result.frames_with_lock += 1

        record = self._build_record(frame_index, detections, tracks, primary, sharpness)
        if self._export_fh is not None:
            self._export_fh.write(json.dumps(record) + "\n")
        return record

    def _motion_diagnostics(self) -> dict:
        """Camera-regime diagnostics, unwrapping fusion's sub-detectors.

        Exported per frame as well as shown on the HUD: mode stability is a
        regression signal in its own right. A camera that is plainly moving but
        is repeatedly declared static means the ego-motion estimate is failing,
        and that silently blinds the detector while the background model
        rebuilds - which is exactly how it failed on real footage before
        ``_estimate_ego_motion`` learned to say "unknown".
        """
        detector = self.detector
        for sub in getattr(self.detector, "detectors", []):
            if hasattr(sub, "last_mode"):
                detector = sub
                break
        return {
            "camera_mode": getattr(detector, "last_mode", "-"),
            "ego_motion": float(getattr(detector, "last_ego_motion", 0.0)),
            "ego_measured": bool(getattr(detector, "last_ego_measured", True)),
            "warming_up": bool(getattr(detector, "warming_up", False)),
        }

    def _build_stats(self, frame_index: int, detections: list, tracks: list[Track]) -> dict:
        track_stats = self.tracker.stats()
        diag = self._motion_diagnostics()
        camera_mode = diag["camera_mode"]
        ego_motion = diag["ego_motion"]

        return {
            "frame_index": frame_index,
            "fps": self.perf.fps,
            "source_fps": self.source.info.fps,
            "source_name": os.path.basename(self.source.info.path),
            "backend": self.detector.name,
            "camera_mode": camera_mode,
            "ego_motion": ego_motion,
            "detections": len(detections),
            "total_tracks": track_stats["total"],
            "confirmed": track_stats["confirmed"],
            "coasting": track_stats["coasting"],
            "timings": {
                "det": self.perf.stage_ms("detect"),
                "trk": self.perf.stage_ms("track"),
                "rnd": self.perf.stage_ms("render"),
            },
        }

    def _build_record(
        self,
        frame_index: int,
        detections: list[Detection],
        tracks: list[Track],
        primary: Track | None,
        sharpness: float,
    ) -> dict:
        fps = self.source.info.fps
        exported = [
            t for t in tracks if (self._export_tentative or t.is_confirmed)
        ]
        diag = self._motion_diagnostics()
        return {
            "frame": frame_index,
            "timestamp": round(frame_index / max(fps, 1e-6), 3),
            "camera_mode": diag["camera_mode"],
            "ego_measured": diag["ego_measured"],
            "warming_up": diag["warming_up"],
            "detections": [d.to_dict() for d in detections],
            "tracks": [t.to_dict(fps) for t in exported],
            "primary_id": primary.id if primary is not None else None,
            "focus": {
                "sharpness": round(float(sharpness), 2),
                "normalised": round(self.focus.normalised, 4),
                "state": self.focus.state,
            },
            "zoom": round(self.viewport.zoom, 3),
        }

    # =============================================================== controls
    def _handle_keys(self) -> bool:
        """Process one key event. Returns False when the run should stop."""
        key = cv2.waitKey(1 if not self._paused else 30) & 0xFF
        if key == 255:
            return True

        if key in (ord("q"), 27):  # q / ESC
            self._quit = True
            return False
        if key == ord(" "):
            self._paused = not self._paused
        elif key in (ord("+"), ord("=")):
            self.viewport.zoom_in()
        elif key in (ord("-"), ord("_")):
            self.viewport.zoom_out()
        elif key == ord("0"):
            self.viewport.reset()
        elif key == ord("t"):
            self.tracker.cycle_primary()
        elif key == ord("y"):
            self.tracker.clear_manual_lock()
        elif key == ord("f"):
            self.focus.assist_enabled = not self.focus.assist_enabled
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("m"):
            self.show_mask = not self.show_mask
        elif key == ord("g"):
            self.hud.show_grid = not self.hud.show_grid
        elif key == ord("d"):
            self.hud.show_detections = not self.hud.show_detections
        elif key == ord("a"):
            self.viewport.auto_follow = not self.viewport.auto_follow
        elif key == ord("s"):
            self._snapshot()
        elif key in (81, ord("j")):   # left arrow
            self.viewport.pan(-24, 0)
        elif key in (83, ord("l")):   # right arrow
            self.viewport.pan(24, 0)
        elif key in (82, ord("i")):   # up arrow
            self.viewport.pan(0, -24)
        elif key in (84, ord("k")):   # down arrow
            self.viewport.pan(0, 24)
        elif key == ord("n") and self._paused:
            self._step_once = True
        return True

    def _snapshot(self) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"snapshot_{int(time.time())}.png")
        frame = getattr(self, "_display_frame", None)
        if frame is not None:
            cv2.imwrite(path, frame)
            log.info("Snapshot written: %s", path)

    # ================================================================ cleanup
    def _cleanup(self) -> None:
        self.source.release()
        if self.recorder is not None:
            self.recorder.release()
        if self._export_fh is not None:
            self._export_fh.close()
            self._export_fh = None
        if self.display:
            cv2.destroyAllWindows()
