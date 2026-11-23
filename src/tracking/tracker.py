"""Multi-object tracker with global assignment and primary-target selection.

Association is solved globally per frame with the Hungarian algorithm rather
than greedily, because greedy matching produces ID swaps exactly in the case
that matters most here - two contacts crossing in the image plane.

Gating happens before the assignment, on two independent criteria:

1. **Mahalanobis gate** - rejects matches the motion model considers
   implausible, taking the filter's own uncertainty into account.
2. **Geometric gate** - IoU, *or* centre distance as a fallback. A small fast
   UAS can move further than its own width between frames, giving IoU of exactly
   zero for the correct association, so IoU alone would break the track and
   spawn a new ID every few frames.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detection.base import Detection
from ..utils.geometry import center_distance_matrix, iou_matrix, xyxy_to_cxcywh
from .track import Track, TrackState

__all__ = ["MultiObjectTracker"]


class MultiObjectTracker:
    """Tracks multiple contacts and nominates one primary (locked) target."""

    def __init__(self, cfg, trail_length: int = 48) -> None:
        self.cfg = cfg
        self.min_hits = int(cfg.get("min_hits", 3))
        self.max_age = int(cfg.get("max_age", 30))
        self.min_iou = float(cfg.get("min_iou", 0.10))
        self.mahalanobis_gate = float(cfg.get("mahalanobis_gate", 9.4877))
        self.max_center_distance = float(cfg.get("max_center_distance", 120.0))
        self.kalman_cfg = dict(cfg.get("kalman", {}) or {})
        self.trail_length = int(trail_length)
        self.coast_floor = float(cfg.get("coast_floor", 0.5))
        self.coast_maturity_hits = float(cfg.get("coast_maturity_hits", 25.0))
        self.min_displacement = float(cfg.get("min_displacement", 6.0))

        priority = dict(cfg.get("priority", {}) or {})
        self._w_conf = float(priority.get("w_confidence", 0.40))
        self._w_persist = float(priority.get("w_persistence", 0.25))
        self._w_size = float(priority.get("w_size", 0.15))
        self._w_central = float(priority.get("w_centrality", 0.20))
        self._switch_margin = float(priority.get("switch_margin", 0.12))
        self._switch_frames = int(priority.get("switch_frames", 8))

        self.tracks: list[Track] = []
        self._primary_id: int | None = None
        self._challenger_id: int | None = None
        self._challenger_frames = 0
        # Operator override: when set, priority scoring is bypassed.
        self._manual_lock_id: int | None = None

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        self.tracks.clear()
        self._primary_id = None
        self._challenger_id = None
        self._challenger_frames = 0
        self._manual_lock_id = None

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame_shape: tuple[int, int],
        detections_valid: bool = True,
    ) -> list[Track]:
        """Advance all tracks by one frame and associate ``detections``.

        Args:
            detections: Contacts found in this frame (may be empty).
            frame_index: Absolute frame number, recorded on tracks.
            frame_shape: ``(height, width)`` - used for centrality scoring.
            detections_valid: ``False`` when the detector was not run for this
                frame at all (``detection.interval`` > 1). An empty list then
                means "not looked at", not "looked and found nothing", and the
                two must not be treated alike - see ``Track.predict``.

        Returns:
            The list of currently active tracks.
        """
        for track in self.tracks:
            track.predict(counts_as_miss=detections_valid)

        if not detections_valid:
            # Extrapolate only. No association, no ageing, no new tracks: there
            # is no evidence this frame to support any of those decisions.
            self._update_primary(frame_shape)
            return self.tracks

        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx], frame_index, self.min_hits)

        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed(self.max_age, self.min_hits, frame_shape)

        for d_idx in unmatched_dets:
            self.tracks.append(
                Track(
                    detections[d_idx],
                    frame_index,
                    kalman_cfg=self.kalman_cfg,
                    trail_length=self.trail_length,
                    coast_floor=self.coast_floor,
                    coast_maturity_hits=self.coast_maturity_hits,
                    min_displacement=self.min_displacement,
                )
            )

        self.tracks = [t for t in self.tracks if t.is_active]
        self._update_primary(frame_shape)
        return self.tracks

    # ------------------------------------------------------------ association
    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(self.tracks))), []

        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        det_measurements = xyxy_to_cxcywh(det_boxes).astype(np.float64)

        ious = iou_matrix(track_boxes, det_boxes)
        distances = center_distance_matrix(track_boxes, det_boxes)

        # Cost is driven by IoU where boxes overlap, and by normalised centre
        # distance where they do not, so both regimes live on one [0,1] scale.
        cost = np.where(
            ious > 0.0,
            1.0 - ious,
            1.0 + np.clip(distances / max(self.max_center_distance, 1e-6), 0.0, 1.0),
        ).astype(np.float64)

        # Build the feasibility mask.
        feasible = (ious >= self.min_iou) | (distances <= self.max_center_distance)
        for t_idx, track in enumerate(self.tracks):
            maha = track.kf.gating_distance(det_measurements)
            feasible[t_idx] &= maha <= self.mahalanobis_gate

        INFEASIBLE = 1e6
        cost[~feasible] = INFEASIBLE

        row_idx, col_idx = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= INFEASIBLE:
                continue  # assignment only filled a hole in the rectangular matrix
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    # --------------------------------------------------------------- priority
    def _priority_score(self, track: Track, frame_shape: tuple[int, int]) -> float:
        h, w = frame_shape
        cx, cy = track.center
        # Normalised distance from frame centre, 0 at centre, 1 at the corner.
        max_dist = float(np.hypot(w * 0.5, h * 0.5)) or 1.0
        centrality = 1.0 - float(np.hypot(cx - w * 0.5, cy - h * 0.5)) / max_dist

        area = float(track.size[0] * track.size[1])
        size_term = float(np.tanh(area / 2500.0))
        persistence = min(1.0, track.hits / 15.0)

        return float(
            self._w_conf * track.confidence
            + self._w_persist * persistence
            + self._w_size * size_term
            + self._w_central * max(0.0, centrality)
        )

    def _update_primary(self, frame_shape: tuple[int, int]) -> None:
        """Choose the locked target, with hysteresis to prevent lock flicker."""
        candidates = [t for t in self.tracks if t.is_confirmed]
        if not candidates:
            self._primary_id = None
            self._challenger_id = None
            self._challenger_frames = 0
            return

        # Manual lock wins as long as the operator's chosen track still lives.
        if self._manual_lock_id is not None:
            if any(t.id == self._manual_lock_id for t in candidates):
                self._primary_id = self._manual_lock_id
                return
            self._manual_lock_id = None  # target lost; fall back to automatic

        scores = {t.id: self._priority_score(t, frame_shape) for t in candidates}
        best_id = max(scores, key=lambda tid: scores[tid])

        if self._primary_id is None or self._primary_id not in scores:
            self._primary_id = best_id
            self._challenger_id = None
            self._challenger_frames = 0
            return

        if best_id == self._primary_id:
            self._challenger_id = None
            self._challenger_frames = 0
            return

        # A challenger must beat the incumbent by a margin, for several frames,
        # before the lock moves. Without this the reticle jitters between two
        # similar contacts every frame and the operator cannot read it.
        if scores[best_id] - scores[self._primary_id] < self._switch_margin:
            self._challenger_id = None
            self._challenger_frames = 0
            return

        if best_id == self._challenger_id:
            self._challenger_frames += 1
        else:
            self._challenger_id = best_id
            self._challenger_frames = 1

        if self._challenger_frames >= self._switch_frames:
            self._primary_id = best_id
            self._challenger_id = None
            self._challenger_frames = 0

    # ---------------------------------------------------------------- queries
    @property
    def primary(self) -> Track | None:
        if self._primary_id is None:
            return None
        return next((t for t in self.tracks if t.id == self._primary_id), None)

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.is_confirmed]

    def cycle_primary(self) -> Track | None:
        """Operator control: manually step the lock to the next confirmed track."""
        candidates = sorted(self.confirmed, key=lambda t: t.id)
        if not candidates:
            return None
        if self._primary_id is None:
            chosen = candidates[0]
        else:
            ids = [t.id for t in candidates]
            if self._primary_id in ids:
                chosen = candidates[(ids.index(self._primary_id) + 1) % len(candidates)]
            else:
                chosen = candidates[0]
        self._manual_lock_id = chosen.id
        self._primary_id = chosen.id
        self._challenger_id = None
        self._challenger_frames = 0
        return chosen

    def clear_manual_lock(self) -> None:
        """Hand target selection back to the automatic priority scorer."""
        self._manual_lock_id = None

    @property
    def manual_lock_active(self) -> bool:
        return self._manual_lock_id is not None

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.tracks),
            "confirmed": sum(1 for t in self.tracks if t.state is TrackState.CONFIRMED),
            "coasting": sum(1 for t in self.tracks if t.state is TrackState.COASTING),
            "tentative": sum(1 for t in self.tracks if t.state is TrackState.TENTATIVE),
        }
