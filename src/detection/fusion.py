"""Multi-source detection fusion.

Motion and appearance fail in opposite situations: a motion detector fires on
anything that moves (birds, rain, warp artefacts) but sees a 12 px target at
range; a CNN classifies confidently but needs enough pixels to do so. Running
both and combining the evidence is what makes the combination worth more than
either alone.

Where a motion blob and a CNN box overlap, confidence is combined as the
probabilistic OR of independent evidence::

    p = 1 - (1 - p_motion)(1 - p_cnn)

so two mediocre but agreeing observations produce a strong contact, while a
single-source detection keeps its original (lower) score and must earn
promotion through temporal consistency in the tracker.
"""

from __future__ import annotations

import numpy as np

from ..utils.geometry import iou_matrix, nms
from .base import Detection, Detector

__all__ = ["FusionDetector"]


class FusionDetector(Detector):
    """Runs several detectors and merges their output into one contact list."""

    name = "fusion"

    def __init__(self, detectors: list[Detector], cfg) -> None:
        if not detectors:
            raise ValueError("FusionDetector needs at least one sub-detector")
        self.detectors = detectors
        self.match_iou = float(cfg.get("match_iou", 0.30))
        self.max_conf = float(cfg.get("max_combined_confidence", 0.97))
        self.nms_iou = float(cfg.get("nms_iou", 0.55))

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        for det in self.detectors:
            det.reset()

    @property
    def debug_mask(self) -> np.ndarray | None:
        for det in self.detectors:
            mask = det.debug_mask
            if mask is not None:
                return mask
        return None

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        per_source: list[list[Detection]] = [
            det.detect(frame, frame_index) for det in self.detectors
        ]
        pool: list[Detection] = [d for group in per_source for d in group]
        if len(pool) <= 1:
            return pool

        merged = self._combine_evidence(per_source)

        boxes = np.array([d.bbox for d in merged], dtype=np.float32)
        scores = np.array([d.score for d in merged], dtype=np.float32)
        keep = nms(boxes, scores, self.nms_iou)
        return [merged[i] for i in keep]

    # -------------------------------------------------------------- internals
    def _combine_evidence(self, per_source: list[list[Detection]]) -> list[Detection]:
        """Fuse pairwise-overlapping detections across sources.

        Each detection from the first source is matched against every later
        source; agreeing pairs are merged into a single fused contact whose box
        is the confidence-weighted mean of the contributors.
        """
        if len(per_source) == 1:
            return list(per_source[0])

        fused: list[Detection] = []
        consumed: list[set[int]] = [set() for _ in per_source]

        primary = per_source[0]
        for i, det in enumerate(primary):
            partners: list[tuple[int, int, Detection]] = []
            for s_idx in range(1, len(per_source)):
                others = per_source[s_idx]
                if not others:
                    continue
                ious = iou_matrix(
                    det.bbox[None, :], np.array([o.bbox for o in others], dtype=np.float32)
                )[0]
                best = int(np.argmax(ious))
                if ious[best] >= self.match_iou and best not in consumed[s_idx]:
                    partners.append((s_idx, best, others[best]))

            if not partners:
                continue

            consumed[0].add(i)
            contributors = [det]
            for s_idx, o_idx, other in partners:
                consumed[s_idx].add(o_idx)
                contributors.append(other)

            fused.append(self._merge(contributors))

        # Anything that found no partner survives on its own merit.
        for s_idx, group in enumerate(per_source):
            for i, det in enumerate(group):
                if i not in consumed[s_idx]:
                    fused.append(det)
        return fused

    def _merge(self, contributors: list[Detection]) -> Detection:
        weights = np.array([max(c.score, 1e-3) for c in contributors], dtype=np.float32)
        boxes = np.array([c.bbox for c in contributors], dtype=np.float32)
        bbox = (boxes * weights[:, None]).sum(axis=0) / weights.sum()

        # Probabilistic OR of independent evidence.
        miss = 1.0
        for c in contributors:
            miss *= (1.0 - c.score)
        score = min(self.max_conf, 1.0 - miss)

        # Prefer a semantic label from the CNN over the generic motion label.
        label = next(
            (c.label for c in contributors if c.source == "cnn"), contributors[0].label
        )
        meta: dict = {"sources": [c.source for c in contributors]}
        for c in contributors:
            for k, v in c.meta.items():
                meta.setdefault(f"{c.source}_{k}", v)

        return Detection(bbox=bbox, score=score, label=label, source=self.name, meta=meta)
