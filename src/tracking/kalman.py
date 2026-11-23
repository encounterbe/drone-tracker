"""Constant-velocity Kalman filter for image-plane box tracking.

State (8-D)::

    x = [cx, cy, w, h, vx, vy, vw, vh]

Measurement (4-D) is the detected box in centre form ``[cx, cy, w, h]``.

Two design choices worth stating explicitly:

*Size-scaled noise.* Process and measurement noise are proportional to the
object's current height rather than constant. A 200 px target that is 3 px off
is well localised; a 12 px target that is 3 px off is badly localised. Absolute
noise terms would treat those identically and either over-trust distant targets
or over-smooth near ones.

*Coasting is a feature.* When detections drop out - the drone crosses a tree
line, or passes in front of clutter - the filter keeps propagating velocity so
the track survives the gap and re-associates on the far side instead of being
deleted and re-born with a new ID.
"""

from __future__ import annotations

import numpy as np

__all__ = ["KalmanBoxFilter"]


class KalmanBoxFilter:
    """Linear Kalman filter over box centre, size and their velocities."""

    def __init__(
        self,
        measurement: np.ndarray,
        position_std: float = 0.05,
        velocity_std: float = 0.012,
        measurement_std: float = 0.06,
    ) -> None:
        """Initialise from a first measurement ``[cx, cy, w, h]``."""
        self._std_pos = float(position_std)
        self._std_vel = float(velocity_std)
        self._std_meas = float(measurement_std)

        ndim = 4
        # Constant-velocity transition: position += velocity, velocity constant.
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = 1.0
        # We observe position/size only, never velocity.
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

        z = np.asarray(measurement, dtype=np.float64).reshape(4)
        self.x = np.concatenate([z, np.zeros(4, dtype=np.float64)])

        # Initial uncertainty: position reasonably known, velocity entirely unknown.
        h = max(float(z[3]), 1.0)
        std = np.array(
            [
                2 * self._std_pos * h,
                2 * self._std_pos * h,
                2 * self._std_pos * h,
                2 * self._std_pos * h,
                10 * self._std_vel * h,
                10 * self._std_vel * h,
                10 * self._std_vel * h,
                10 * self._std_vel * h,
            ],
            dtype=np.float64,
        )
        self.P = np.diag(np.square(std))

    # ----------------------------------------------------------------- noise
    def _process_noise(self) -> np.ndarray:
        h = max(float(self.x[3]), 1.0)
        std = np.array(
            [
                self._std_pos * h,
                self._std_pos * h,
                self._std_pos * h,
                self._std_pos * h,
                self._std_vel * h,
                self._std_vel * h,
                self._std_vel * h,
                self._std_vel * h,
            ],
            dtype=np.float64,
        )
        return np.diag(np.square(std))

    def _measurement_noise(self) -> np.ndarray:
        h = max(float(self.x[3]), 1.0)
        std = np.full(4, self._std_meas * h, dtype=np.float64)
        return np.diag(np.square(std))

    # ------------------------------------------------------------- filtering
    def predict(self, freeze_size: bool = False) -> np.ndarray:
        """Advance one time step. Returns the predicted state.

        ``freeze_size`` zeroes the width/height velocities before propagating.
        Use it while coasting: position extrapolates from real evidence about
        where the object is going, but *size* change is an artefact of the last
        few noisy measurements, and extrapolating it through a long gap drives
        the box to the 1 px floor. A degenerate box is not just cosmetic - it has
        zero IoU with every detection, so the track cannot re-acquire its own
        target when detections come back.
        """
        if freeze_size:
            self.x[6] = 0.0
            self.x[7] = 0.0
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._process_noise()
        # Size must stay positive even under aggressive coasting.
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.x

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        """Project state into measurement space: ``(mean, covariance)``."""
        mean = self._H @ self.x
        cov = self._H @ self.P @ self._H.T + self._measurement_noise()
        return mean, cov

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Correct the state with a measurement ``[cx, cy, w, h]``."""
        z = np.asarray(measurement, dtype=np.float64).reshape(4)
        mean, S = self.project()

        # Solve for the gain rather than inverting S explicitly: K = P H^T S^-1
        # becomes (S^T \ (P H^T)^T)^T, which is better conditioned.
        PHt = self.P @ self._H.T
        try:
            K = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate covariance
            K = PHt @ np.linalg.pinv(S)

        innovation = z - mean
        self.x = self.x + K @ innovation

        # Joseph-free but symmetrised covariance update, guarded against drift.
        identity = np.eye(self.P.shape[0])
        self.P = (identity - K @ self._H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.x

    # -------------------------------------------------------------- distance
    def gating_distance(
        self, measurements: np.ndarray, position_only: bool = True
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the prediction to each measurement.

        Compared against a chi-square threshold to reject associations that are
        geometrically implausible regardless of how good their IoU is.

        ``position_only`` (default) gates on centre position alone, 2 degrees of
        freedom. Including width and height sounds stricter and better, but it
        is actively harmful for the targets that matter most here: a faint,
        barely-resolved UAS thresholds into a blob whose apparent size swings
        from one frame to the next, so the size terms dominate the distance and
        veto associations whose position agrees perfectly. The track then dies
        and the same object is reborn under a new ID. Position is the quantity
        the motion model actually predicts well; size is not.
        """
        z = np.atleast_2d(np.asarray(measurements, dtype=np.float64))
        if z.size == 0:
            return np.zeros(0, dtype=np.float64)

        mean, S = self.project()
        if position_only:
            mean = mean[:2]
            S = S[:2, :2]
            z = z[:, :2]
        diff = z - mean[None, :]
        try:
            solved = np.linalg.solve(S, diff.T).T
        except np.linalg.LinAlgError:  # pragma: no cover
            solved = diff @ np.linalg.pinv(S)
        return np.einsum("ij,ij->i", diff, solved)

    # ------------------------------------------------------------ accessors
    @property
    def center(self) -> np.ndarray:
        return self.x[:2].astype(np.float32)

    @property
    def size(self) -> np.ndarray:
        return self.x[2:4].astype(np.float32)

    @property
    def velocity(self) -> np.ndarray:
        """Centre velocity in px/frame."""
        return self.x[4:6].astype(np.float32)

    @property
    def position_uncertainty(self) -> float:
        """RMS positional standard deviation in px - drives the HUD error ellipse."""
        return float(np.sqrt(max(self.P[0, 0], 0.0) + max(self.P[1, 1], 0.0)))
