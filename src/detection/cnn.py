"""Neural detection backend (Ultralytics YOLO).

This is an *optional* backend. ``ultralytics`` is not a hard dependency, and the
default COCO weights contain no "drone" class - see the honest caveat below. If
the package or the weights are missing, construction raises
``DetectorUnavailable`` and the pipeline falls back to motion detection rather
than failing.

Class-mapping caveat
--------------------
A stock ``yolov8n.pt`` is trained on COCO, which has no UAS category. Out of the
box this backend therefore treats a small set of COCO classes that a quadcopter
is commonly *confused with* - ``airplane``, ``bird``, ``kite`` - as generic
airborne contacts, and relabels them ``AIR-CONTACT``. That is a stand-in to make
the plumbing testable, **not** a drone detector: expect it to miss real drones
and to fire on real birds.

For actual performance you must train on UAS imagery and point ``weights`` at
your own ``.pt`` file. docs/ARCHITECTURE.md documents the datasets and the training recipe.
"""

from __future__ import annotations

import os

import numpy as np

from .base import Detection, Detector, DetectorUnavailable

__all__ = ["CNNDetector"]


class CNNDetector(Detector):
    """Ultralytics YOLO wrapper producing ``Detection`` objects."""

    name = "cnn"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.weights = str(cfg.get("weights", "yolov8n.pt"))
        self.confidence = float(cfg.get("confidence", 0.25))
        self.iou = float(cfg.get("iou", 0.45))
        self.device = str(cfg.get("device", "cpu"))
        self.imgsz = int(cfg.get("imgsz", 960))
        self.remap_label = str(cfg.get("remap_label", "AIR-CONTACT"))

        proxy = cfg.get("coco_proxy_classes", [4, 14, 33])
        self.class_filter: set[int] | None = set(int(c) for c in proxy) if proxy else None

        try:
            from ultralytics import YOLO  # noqa: PLC0415 - intentionally lazy
        except ImportError as exc:
            raise DetectorUnavailable(
                "The 'cnn' backend needs the ultralytics package: pip install ultralytics"
            ) from exc

        # A bare model name (e.g. "yolov8n.pt") is auto-downloaded by ultralytics;
        # an explicit path that does not exist is a user error worth reporting.
        if os.sep in self.weights and not os.path.exists(self.weights):
            raise DetectorUnavailable(f"Weights file not found: {self.weights}")

        try:
            self._model = YOLO(self.weights)
        except Exception as exc:  # network failure, corrupt weights, ...
            raise DetectorUnavailable(f"Could not load weights {self.weights!r}: {exc}") from exc

        self._names: dict[int, str] = dict(getattr(self._model, "names", {}) or {})
        # A model whose classes are not COCO is assumed purpose-trained: in that
        # case every class it reports is meaningful, so drop the proxy filter.
        if self._names and len(self._names) != 80:
            self.class_filter = None

    # ------------------------------------------------------------------ API
    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        classes = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for bbox, score, cls_id in zip(xyxy, scores, classes):
            if self.class_filter is not None and int(cls_id) not in self.class_filter:
                continue
            raw_name = self._names.get(int(cls_id), str(int(cls_id)))
            label = self.remap_label if self.class_filter is not None else raw_name.upper()
            detections.append(
                Detection(
                    bbox=bbox,
                    score=float(score),
                    label=label,
                    source=self.name,
                    meta={"class_id": int(cls_id), "class_name": raw_name},
                )
            )
        return detections
