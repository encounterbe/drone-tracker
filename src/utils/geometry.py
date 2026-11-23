"""Bounding-box geometry helpers.

All boxes are numpy float32 arrays in ``xyxy`` order (x1, y1, x2, y2) with
image-pixel units unless stated otherwise. Functions are vectorised where it
matters; per-frame detection counts are small, but the tracker's cost matrix
is built every frame and benefits from array maths.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "box_area",
    "iou_matrix",
    "center_distance_matrix",
    "clip_boxes",
    "expand_box",
    "nms",
]


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert (..., 4) boxes from corner form to centre form."""
    boxes = np.asarray(boxes, dtype=np.float32)
    single = boxes.ndim == 1
    b = np.atleast_2d(boxes)
    out = np.empty_like(b)
    out[:, 0] = (b[:, 0] + b[:, 2]) * 0.5
    out[:, 1] = (b[:, 1] + b[:, 3]) * 0.5
    out[:, 2] = b[:, 2] - b[:, 0]
    out[:, 3] = b[:, 3] - b[:, 1]
    return out[0] if single else out


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (..., 4) boxes from centre form to corner form."""
    boxes = np.asarray(boxes, dtype=np.float32)
    single = boxes.ndim == 1
    b = np.atleast_2d(boxes)
    out = np.empty_like(b)
    half_w = b[:, 2] * 0.5
    half_h = b[:, 3] * 0.5
    out[:, 0] = b[:, 0] - half_w
    out[:, 1] = b[:, 1] - half_h
    out[:, 2] = b[:, 0] + half_w
    out[:, 3] = b[:, 1] + half_h
    return out[0] if single else out


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of each box; negative-size boxes clamp to zero."""
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes.

    Returns an ``(len(a), len(b))`` matrix. Empty inputs yield a correctly
    shaped empty array so callers never need to special-case them.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a) if a.size else 0, len(b) if b.size else 0), dtype=np.float32)

    # Broadcast to (Na, Nb, ...) for the intersection rectangle.
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = box_area(a)[:, None]
    area_b = box_area(b)[None, :]
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def center_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between box centres, in pixels.

    Needed because a small fast-moving UAS can travel further than its own
    width between frames, making IoU zero even for a correct association.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a) if a.size else 0, len(b) if b.size else 0), dtype=np.float32)
    ca = xyxy_to_cxcywh(a)[:, :2]
    cb = xyxy_to_cxcywh(b)[:, :2]
    return np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2).astype(np.float32)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp boxes to the image rectangle."""
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float32)).copy()
    b[:, 0] = np.clip(b[:, 0], 0, width - 1)
    b[:, 1] = np.clip(b[:, 1], 0, height - 1)
    b[:, 2] = np.clip(b[:, 2], 0, width - 1)
    b[:, 3] = np.clip(b[:, 3], 0, height - 1)
    return b


def expand_box(box: np.ndarray, margin: float, width: int, height: int) -> np.ndarray:
    """Grow a box by ``margin`` px on every side, clipped to the image."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.array(
        [
            max(0.0, x1 - margin),
            max(0.0, y1 - margin),
            min(width - 1.0, x2 + margin),
            min(height - 1.0, y2 + margin),
        ],
        dtype=np.float32,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float32))
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.size == 0:
        return []

    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[best][None, :], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep
