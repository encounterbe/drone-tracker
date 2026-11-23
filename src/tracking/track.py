"""Track object and lifecycle state machine.

States
------
``TENTATIVE``
    Newly created from an unmatched detection. Not shown as a confirmed contact
    and not eligible to be the primary target. This is the main false-positive
    filter: noise blobs rarely survive ``min_hits`` consecutive associations.

``CONFIRMED``
    Associated with detections often enough to be believed. Drawn on the HUD and
    eligible for target lock.

``COASTING``
    Was confirmed, currently has no detection. The Kalman filter propagates the
    motion model; the HUD marks it as extrapolated so an operator never mistakes
    a prediction for an observation.

``LOST``
    Terminal. Exceeded ``max_age`` without re-association; removed by the tracker.
"""

from __future__ import annotations

import itertools
from collections import deque
from enum import Enum
from typing import Any

import numpy as np

from ..detection.base import Detection
from ..utils.geometry import cxcywh_to_xyxy, xyxy_to_cxcywh
from .kalman import KalmanBoxFilter

__all__ = ["Track", "TrackState"]


class TrackState(Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    LOST = "LOST"


class Track:
    """One hypothesised object, persisting across frames."""

    _id_counter = itertools.count(1)

    def __init__(
        self,
        detection: Detection,
        frame_index: int,
        kalman_cfg: dict | None = None,
        trail_length: int = 48,
        coast_floor: float = 0.5,
        coast_maturity_hits: float = 25.0,
        min_displacement: float = 6.0,
    ) -> None:
        self.id: int = next(Track._id_counter)
        kalman_cfg = kalman_cfg or {}
        self._coast_floor = float(coast_floor)
        self._coast_maturity_hits = float(coast_maturity_hits)
        self._min_displacement = float(min_displacement)

        self.kf = KalmanBoxFilter(
            xyxy_to_cxcywh(detection.bbox).astype(np.float64),
            position_std=float(kalman_cfg.get("position_std", 0.05)),
            velocity_std=float(kalman_cfg.get("velocity_std", 0.012)),
            measurement_std=float(kalman_cfg.get("measurement_std", 0.06)),
        )

        self.state = TrackState.TENTATIVE
        self.label = detection.label
        self.source = detection.source

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.first_frame = int(frame_index)
        self.last_frame = int(frame_index)

        # EMA of detection confidence - smoother than the latest frame's value.
        self.confidence = float(detection.score)
        self._conf_alpha = 0.35
        # Per-frame confidence decay applied while coasting.
        self._coast_decay = 0.94

        self.trail: deque[tuple[float, float]] = deque(maxlen=int(trail_length))
        self.trail.append(tuple(self.center))

        # Extent of positions visited, for the displacement gate (see
        # displacement / update). Kept as a running min/max rather than derived
        # from `trail`, which is bounded and would forget early positions.
        self._pos_min = self.center.astype(np.float64).copy()
        self._pos_max = self.center.astype(np.float64).copy()

        # Rolling sharpness measurements from the focus module, for the HUD.
        self.sharpness_history: deque[float] = deque(maxlen=60)
        self.meta: dict[str, Any] = dict(detection.meta)

    # ------------------------------------------------------------- lifecycle
    def predict(self, counts_as_miss: bool = True) -> None:
        """Advance the motion model by one frame.

        ``counts_as_miss=False`` is for frames where the detector was never run
        (``detection.interval`` > 1). The position is still extrapolated, so the
        box follows the target between detections, but the track is not aged
        toward death - nothing was missed, because nothing was attempted.

        Conflating the two made ``detection.interval`` unusable: on every
        skipped frame `mark_missed` ran, and a tentative track that misses once
        is deleted as noise, so no track could ever accumulate ``min_hits``
        between two detection frames. Measured on the fpv clip, interval 3 gave
        **0 % recall** - the throughput setting advertised as "roughly doubles
        it" silently detected nothing at all.
        """
        self.kf.predict(freeze_size=self.time_since_update > 0)
        self.age += 1
        if not counts_as_miss:
            return
        self.time_since_update += 1
        if self.state is TrackState.CONFIRMED and self.time_since_update > 0:
            self.state = TrackState.COASTING

    def update(self, detection: Detection, frame_index: int, min_hits: int) -> None:
        """Associate a detection with this track and correct the state."""
        self.kf.update(xyxy_to_cxcywh(detection.bbox).astype(np.float64))
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = int(frame_index)
        self.label = detection.label
        self.source = detection.source
        self.meta = dict(detection.meta)

        self.confidence = (
            self._conf_alpha * float(detection.score) + (1 - self._conf_alpha) * self.confidence
        )
        self.trail.append(tuple(self.center))

        center = self.center.astype(np.float64)
        self._pos_min = np.minimum(self._pos_min, center)
        self._pos_max = np.maximum(self._pos_max, center)

        if self.state in (TrackState.TENTATIVE, TrackState.COASTING):
            if self.state is TrackState.COASTING:
                self.state = TrackState.CONFIRMED
            elif self.hits >= min_hits and self.displacement >= self._min_displacement:
                self.state = TrackState.CONFIRMED

    def mark_missed(
        self, max_age: int, min_hits: int, frame_shape: tuple[int, int] | None = None
    ) -> None:
        """Handle a frame with no associated detection.

        Coasting exists to survive *occlusion* - the target passing behind a
        tree line or a cloud - and nothing else. A track whose prediction has
        left the sensor's field of view has departed, not been occluded: it can
        never be re-associated at that position, so extrapolating it for another
        second only litters the display with phantom contacts. Such tracks are
        therefore terminated immediately.
        """
        # A tentative track that misses even once is almost always noise.
        if self.state is TrackState.TENTATIVE and self.hits < min_hits:
            self.state = TrackState.LOST
            return

        if frame_shape is not None and self._has_left_frame(frame_shape):
            self.state = TrackState.LOST
            return

        if self.time_since_update > self.coast_budget(max_age):
            self.state = TrackState.LOST
            return

        # Confidence decays while unsupported, so a long-coasting track cannot
        # hold the operator's target lock against a freshly detected contact.
        self.confidence *= self._coast_decay
        self.trail.append(tuple(self.center))

    def coast_budget(self, max_age: int) -> int:
        """How many frames this track may coast, scaled by how well it is established.

        Willingness to extrapolate should track confidence in the object's
        existence. A contact observed many times has earned the full occlusion
        budget; one that was confirmed three frames ago and immediately vanished
        has not, and extrapolating it for a full second just paints a phantom
        that drifts further from reality every frame.

        The floor matters more than it looks. Faint targets are detected
        *intermittently*, so a young track on a distant UAS routinely sees gaps
        of several frames. Too small a floor deletes it during the gap, and the
        target comes back as a new ID - which reads on the display as the system
        losing and reacquiring the contact over and over. Phantom tracks from
        departed objects are handled separately by the out-of-frame check, so
        the floor can afford to be generous.
        """
        maturity = min(1.0, self.hits / max(1.0, self._coast_maturity_hits))
        budget = max_age * (self._coast_floor + (1.0 - self._coast_floor) * maturity)
        return int(max(4, round(budget)))

    def _has_left_frame(self, frame_shape: tuple[int, int]) -> bool:
        """Whether the predicted centre has moved outside the image.

        A small margin is allowed so a target grazing the edge is not dropped
        the instant its centroid estimate wobbles past the boundary.
        """
        h, w = frame_shape
        cx, cy = self.center
        margin = 0.02 * max(w, h)
        return bool(cx < -margin or cy < -margin or cx > w + margin or cy > h + margin)

    @property
    def is_active(self) -> bool:
        return self.state is not TrackState.LOST

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    @property
    def is_coasting(self) -> bool:
        return self.state is TrackState.COASTING

    # -------------------------------------------------------------- geometry
    @property
    def center(self) -> np.ndarray:
        return self.kf.center

    @property
    def bbox(self) -> np.ndarray:
        """Current estimate as ``[x1, y1, x2, y2]``."""
        return cxcywh_to_xyxy(np.concatenate([self.kf.center, self.kf.size]))

    @property
    def size(self) -> np.ndarray:
        return self.kf.size

    @property
    def velocity(self) -> np.ndarray:
        """Centre velocity in px/frame."""
        return self.kf.velocity

    def speed_px_per_s(self, fps: float) -> float:
        return float(np.linalg.norm(self.velocity) * max(fps, 1e-6))

    def heading_deg(self) -> float:
        """Direction of travel in degrees, 0 = right, 90 = up (screen coords)."""
        vx, vy = self.velocity
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(-vy, vx)) % 360.0)

    @property
    def displacement(self) -> float:
        """Diagonal of the region this track has covered, in pixels.

        Moving-target indication: a contact that has not moved is not an
        aircraft. Measured on the clean test clip, every false contact that
        survived detection-level filtering came from a *stationary* source -
        residual left at a high-contrast cloud edge by an imperfect background
        model - with a total displacement under 3 px, while real tracks covered
        151 px (median). Gating confirmation on displacement removed 100 % of
        those false contacts and 0 % of the real ones.

        Caveat: a genuinely hovering target at long range, moving less than the
        threshold, will not confirm. The gate is a *delay*, not a permanent
        veto - the track confirms as soon as it has moved far enough - but for a
        true station-keeping hover, lower ``tracking.min_displacement``.
        """
        return float(np.linalg.norm(self._pos_max - self._pos_min))

    @property
    def uncertainty(self) -> float:
        return self.kf.position_uncertainty

    @property
    def track_quality(self) -> float:
        """Composite [0,1] quality: confidence tempered by how well it is held."""
        persistence = min(1.0, self.hits / 12.0)
        freshness = 1.0 / (1.0 + 0.25 * self.time_since_update)
        return float(np.clip(0.5 * self.confidence + 0.3 * persistence + 0.2 * freshness, 0, 1))

    @property
    def mean_sharpness(self) -> float:
        if not self.sharpness_history:
            return 0.0
        return float(sum(self.sharpness_history) / len(self.sharpness_history))

    # ----------------------------------------------------------------- export
    def to_dict(self, fps: float = 30.0) -> dict[str, Any]:
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        cx, cy = (float(v) for v in self.center)
        return {
            "id": self.id,
            "state": self.state.value,
            "label": self.label,
            "source": self.source,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "center": [round(cx, 2), round(cy, 2)],
            "velocity_px_per_frame": [round(float(v), 3) for v in self.velocity],
            "speed_px_per_s": round(self.speed_px_per_s(fps), 2),
            "heading_deg": round(self.heading_deg(), 1),
            "confidence": round(self.confidence, 4),
            "quality": round(self.track_quality, 4),
            "uncertainty_px": round(self.uncertainty, 2),
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        cx, cy = self.center
        return (
            f"Track(id={self.id}, state={self.state.value}, "
            f"center=({cx:.0f},{cy:.0f}), hits={self.hits}, conf={self.confidence:.2f})"
        )
