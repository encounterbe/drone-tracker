"""Video I/O abstraction.

A single ``VideoSource`` type covers both offline files and live cameras so the
pipeline never branches on where frames come from. That is what makes the jump
from "analyse this MP4" to "run on the live USB camera" a one-flag change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["VideoSource", "VideoRecorder", "SourceInfo"]


@dataclass(frozen=True)
class SourceInfo:
    """Static metadata about an opened source."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int  # 0 for live sources / unknown
    is_live: bool

    @property
    def duration(self) -> float:
        """Length in seconds, or 0.0 when unknown."""
        if self.frame_count <= 0 or self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


class VideoSource:
    """Iterable frame source over a file path or camera index.

    Yields ``(frame_index, frame)`` tuples. Frames are optionally downscaled to
    a target width so that processing cost is decoupled from source resolution,
    and so detector tuning transfers between a 4K clip and a 720p camera.
    """

    def __init__(
        self,
        source: str | int,
        process_width: int = 0,
        loop: bool = False,
        start_frame: int = 0,
        max_frames: int = 0,
    ) -> None:
        self.source = source
        self.process_width = int(process_width)
        self.loop = bool(loop)
        self.start_frame = max(0, int(start_frame))
        self.max_frames = max(0, int(max_frames))

        is_index = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        if is_index:
            self._handle = cv2.VideoCapture(int(source))
            is_live = True
            path = f"camera:{source}"
        else:
            path = str(source)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video source not found: {path}")
            self._handle = cv2.VideoCapture(path)
            is_live = False

        if not self._handle.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        native_w = int(self._handle.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        native_h = int(self._handle.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(self._handle.get(cv2.CAP_PROP_FPS)) or 0.0
        count = int(self._handle.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
        if count < 0:
            count = 0

        # Work out the resolution frames will actually be delivered at.
        out_w, out_h = native_w, native_h
        if self.process_width > 0 and native_w > 0 and native_w != self.process_width:
            scale = self.process_width / float(native_w)
            out_w = self.process_width
            out_h = int(round(native_h * scale))
        self._scale_to = (out_w, out_h) if (out_w, out_h) != (native_w, native_h) else None

        self.info = SourceInfo(
            path=path,
            width=out_w,
            height=out_h,
            fps=fps if fps > 0 else 30.0,
            frame_count=count,
            is_live=is_live,
        )
        self.native_size = (native_w, native_h)

        if self.start_frame and not is_live:
            self._handle.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self._index = self.start_frame
        self._emitted = 0

    # ------------------------------------------------------------------ iter
    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self.max_frames and self._emitted >= self.max_frames:
            raise StopIteration

        ok, frame = self._handle.read()
        if not ok:
            if self.loop and not self.info.is_live:
                self._handle.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self._index = self.start_frame
                ok, frame = self._handle.read()
            if not ok:
                raise StopIteration

        if self._scale_to is not None:
            frame = cv2.resize(frame, self._scale_to, interpolation=cv2.INTER_AREA)

        index = self._index
        self._index += 1
        self._emitted += 1
        return index, frame

    # --------------------------------------------------------------- control
    def seek(self, frame_index: int) -> bool:
        """Jump to an absolute frame index. No-op for live sources."""
        if self.info.is_live:
            return False
        ok = self._handle.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        if ok:
            self._index = max(0, int(frame_index))
        return bool(ok)

    def release(self) -> None:
        if self._handle is not None:
            self._handle.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoRecorder:
    """Writes annotated frames to disk, sized from the first frame it receives."""

    def __init__(self, path: str, fps: float, fourcc: str = "mp4v") -> None:
        self.path = path
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self._size
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"Could not open VideoWriter for {self.path} "
                    f"(codec {self.fourcc!r}); try a different export.fourcc"
                )
        # Guard against a mid-stream size change (e.g. layout toggled by a key).
        if self._size is not None and (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
