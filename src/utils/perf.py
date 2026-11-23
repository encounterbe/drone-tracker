"""Performance instrumentation.

The HUD shows a live breakdown of where frame time goes (capture / detect /
track / render). That is not decoration: on this kind of pipeline the first
thing you need to know when throughput drops is *which* stage regressed.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager

__all__ = ["PerfMeter"]


class PerfMeter:
    """Rolling-window timing for named pipeline stages.

    Usage::

        perf = PerfMeter(window=60)
        with perf.stage("detect"):
            detections = detector.detect(frame, idx)
        perf.tick()                  # call once per completed frame
        print(perf.fps, perf.stage_ms("detect"))
    """

    def __init__(self, window: int = 60) -> None:
        self.window = int(window)
        self._stage_samples: dict[str, deque[float]] = {}
        self._frame_times: deque[float] = deque(maxlen=self.window)
        self._last_tick: float | None = None
        self._frames = 0
        self._t_start = time.perf_counter()

    # ---------------------------------------------------------------- stages
    @contextmanager
    def stage(self, name: str):
        """Time a named block and record it in the rolling window."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            samples = self._stage_samples.get(name)
            if samples is None:
                samples = deque(maxlen=self.window)
                self._stage_samples[name] = samples
            samples.append(elapsed_ms)

    def stage_ms(self, name: str) -> float:
        """Mean duration of a stage over the rolling window, in milliseconds."""
        samples = self._stage_samples.get(name)
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def stages(self) -> dict[str, float]:
        """All stage means, in insertion order."""
        return {name: self.stage_ms(name) for name in self._stage_samples}

    # ---------------------------------------------------------------- frames
    def tick(self) -> None:
        """Mark the end of one processed frame."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._frame_times.append(now - self._last_tick)
        self._last_tick = now
        self._frames += 1

    @property
    def fps(self) -> float:
        """Instantaneous FPS over the rolling window."""
        if not self._frame_times:
            return 0.0
        mean_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / mean_dt if mean_dt > 1e-9 else 0.0

    @property
    def frame_ms(self) -> float:
        """Mean wall-clock time per frame over the rolling window."""
        if not self._frame_times:
            return 0.0
        return 1000.0 * sum(self._frame_times) / len(self._frame_times)

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def elapsed(self) -> float:
        """Seconds since the meter was created."""
        return time.perf_counter() - self._t_start

    @property
    def mean_fps(self) -> float:
        """Average FPS across the whole run (not just the window)."""
        e = self.elapsed
        return self._frames / e if e > 1e-9 else 0.0

    def summary(self) -> str:
        parts = [f"{self._frames} frames in {self.elapsed:.1f}s ({self.mean_fps:.1f} FPS avg)"]
        for name, ms in self.stages().items():
            parts.append(f"{name}={ms:.1f}ms")
        return " | ".join(parts)
