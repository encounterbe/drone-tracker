"""Pan/tilt gimbal control over a serial link (Stage 3).

This converts the tracker's existing target-offset signal into servo commands.
No other module changes: ``tracker.primary`` already exposes exactly the
quantity a gimbal controller consumes, which is why the architecture was kept
the way it is.

Wire protocol (ASCII, newline-terminated, 115200 baud)::

    "<pan>,<tilt>,<laser>\\n"     pan/tilt in [-1000, 1000], laser always 0
    "PING\\n"                      keep-alive, answered with "PONG"
    "STOP\\n"                      immediate centre + failsafe

The firmware re-centres and reports ``EVT FAILSAFE`` if it hears nothing for
one second. That is the important safety property and it lives on the device
deliberately: a crash, an unplugged USB cable or a hung Python process must not
leave the servos driving.

**The laser output is intentionally not wired.** The protocol carries the field
because the original firmware did, and the firmware answers
``ERR LASER_NOT_WIRED`` to any non-zero value. This is a legal question, not a
technical one: directing a laser at aircraft is a criminal offence in Germany
(§ 315 StGB and Luftverkehrsgesetz provisions), and a tracker that is slightly
off - or that locks onto a manned aircraft instead of the intended drone -
points it somewhere nobody chose. The defensible build drives the camera,
keeps the target framed and logs the track. Do not connect the lock signal to a
laser driver without talking to the owner first.
"""

from __future__ import annotations

import logging
import time

import numpy as np

__all__ = ["GimbalController", "GimbalUnavailable"]

log = logging.getLogger(__name__)


class GimbalUnavailable(RuntimeError):
    """Raised when the serial port cannot be opened."""


class PID:
    """Minimal PID with integral clamping.

    The integral term is clamped rather than unbounded because the error here
    is a pixel offset that saturates: once the target leaves the frame the
    error stops growing but does not go away, and an unclamped integral would
    wind up and slam the servo to its limit the moment the target reappears.
    """

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self._integral = 0.0
        self._previous = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._previous = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self._integral = float(
            np.clip(self._integral + error * dt, -self.integral_limit, self.integral_limit)
        )
        derivative = (error - self._previous) / dt
        self._previous = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class GimbalController:
    """Turns primary-target offset into pan/tilt commands."""

    def __init__(self, cfg, frame_size: tuple[int, int]) -> None:
        """Args:
        cfg: The ``control.gimbal`` config node.
        frame_size: ``(width, height)`` of the *source* frame. Offsets are
            computed in source-pixel coordinates, never display coordinates -
            zoom is an operator aid and must not steer the hardware.
        """
        self.cfg = cfg
        self.width, self.height = frame_size
        self.port = str(cfg.get("port", "COM3"))
        self.baud = int(cfg.get("baud", 115200))
        self.invert_pan = bool(cfg.get("invert_pan", False))
        self.invert_tilt = bool(cfg.get("invert_tilt", True))
        self.deadzone_px = float(cfg.get("deadzone_px", 12.0))
        self.max_rate = float(cfg.get("max_command", 1000.0))
        self.send_hz = float(cfg.get("send_hz", 50.0))

        pid_cfg = dict(cfg.get("pid", {}) or {})
        self.pan_pid = PID(
            pid_cfg.get("kp", 1.6), pid_cfg.get("ki", 0.08),
            pid_cfg.get("kd", 0.12), pid_cfg.get("integral_limit", 250.0),
        )
        self.tilt_pid = PID(
            pid_cfg.get("kp", 1.6), pid_cfg.get("ki", 0.08),
            pid_cfg.get("kd", 0.12), pid_cfg.get("integral_limit", 250.0),
        )

        try:
            import serial  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:
            raise GimbalUnavailable(
                "gimbal control needs pyserial: pip install pyserial"
            ) from exc

        try:
            self._link = serial.Serial(self.port, self.baud, timeout=0)
        except Exception as exc:
            raise GimbalUnavailable(f"could not open {self.port!r}: {exc}") from exc

        self._last_send = 0.0
        self._last_time = time.perf_counter()
        self.last_command: tuple[int, int] = (0, 0)
        self.engaged = False

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._link.close()

    def stop(self) -> None:
        """Centre the gimbal and let the firmware fall into failsafe."""
        self._write("STOP")
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.engaged = False
        self.last_command = (0, 0)

    def update(self, primary) -> tuple[int, int] | None:
        """Send one command for the current primary target.

        ``None`` means nothing was sent this call - either the send interval has
        not elapsed, or there is no target worth driving toward.

        A coasting track is deliberately not followed. Coasting is an
        extrapolation, and chasing an extrapolated position moves the camera
        away from where the target actually was last seen, which is the worst
        possible place to be when it reappears.
        """
        now = time.perf_counter()
        if now - self._last_send < 1.0 / max(self.send_hz, 1e-6):
            return None
        dt = now - self._last_time
        self._last_time = now
        self._last_send = now

        if primary is None or primary.is_coasting:
            if self.engaged:
                self.stop()
            else:
                self._write("PING")
            return None

        cx, cy = (float(v) for v in primary.center)
        error_x = cx - self.width * 0.5
        error_y = cy - self.height * 0.5

        if abs(error_x) < self.deadzone_px and abs(error_y) < self.deadzone_px:
            # Inside the deadzone the servos would hunt around the centre and
            # buzz without improving framing.
            self._write("PING")
            return None

        pan = self.pan_pid.step(error_x / (self.width * 0.5), dt)
        tilt = self.tilt_pid.step(error_y / (self.height * 0.5), dt)

        if self.invert_pan:
            pan = -pan
        if self.invert_tilt:
            tilt = -tilt

        pan_cmd = int(np.clip(pan * self.max_rate, -self.max_rate, self.max_rate))
        tilt_cmd = int(np.clip(tilt * self.max_rate, -self.max_rate, self.max_rate))

        # The laser field is always 0 - see the module docstring.
        self._write(f"{pan_cmd},{tilt_cmd},0")
        self.engaged = True
        self.last_command = (pan_cmd, tilt_cmd)
        return self.last_command

    def _write(self, line: str) -> None:
        try:
            self._link.write((line + "\n").encode("ascii"))
        except Exception as exc:
            log.warning("gimbal write failed: %s", exc)
