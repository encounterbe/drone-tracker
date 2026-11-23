#!/usr/bin/env python3
"""Score a track export against *real* footage.

Why not ``tools/evaluate.py``
-----------------------------
``evaluate.py`` assumes ground truth for every frame, because the synthetic
generator draws the target and therefore knows where it is in all of them. Real
footage breaks both halves of that assumption:

* Ground truth is *sparse or absent*. ``evaluate.py`` counts a confirmed track in
  an unlabelled frame as a false positive, so pointing it at a partially
  labelled clip reports a precision near zero regardless of how the system did.
* Some of the most useful real clips contain **no target at all**. A clip of an
  empty urban street is not a degenerate case to be tolerated - it is the
  single most informative test available, because every contact the system
  reports on it is provably false. No labelling effort at all, and it measures
  exactly the quantity that decides whether an operator will keep the system
  switched on.

So this tool has two modes:

``--expect-none``
    False-alarm mode. Ground truth is "nothing airborne". Reports false
    confirmed tracks per minute, false-lock rate and detection load. Lower is
    better; zero is the goal.

``--ground-truth FILE``
    Sparse mode. Only frames that carry a label are scored, so unlabelled
    stretches neither help nor hurt.

Usage::

    python -m src.main --source video/test.mp4 --no-display --export-tracks /tmp/t.jsonl
    python tools/eval_real.py --tracks /tmp/t.jsonl --expect-none --fps 29.97

    python tools/eval_real.py --tracks /tmp/t3.jsonl \\
        --ground-truth output/test3_groundtruth.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 1e-6 else 0.0


def _center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _confirmed(record: dict) -> list[dict]:
    return [t for t in record["tracks"] if t["state"] in ("CONFIRMED", "COASTING")]


def mode_stability(records: list[dict]) -> dict:
    """Camera-regime health, which needs no ground truth to be informative.

    Two numbers, both of which were badly wrong on real footage:

    ``mode_switches``
        A physically static or continuously moving camera should produce ~1.
        Many switches mean the regime estimate is oscillating, and every switch
        into static throws the background model away.
    ``blind_frac``
        Fraction of frames where the detector emitted nothing because a
        background model was still settling. Start-up costs one warm-up; any
        more than that is a rebuild that should not have happened.
    """
    modes = [r.get("camera_mode") for r in records if r.get("camera_mode")]
    if not modes:
        return {}
    switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
    blind = sum(1 for r in records if r.get("warming_up"))
    unmeasured = sum(1 for r in records if r.get("ego_measured") is False)
    return {
        "mode_switches": switches,
        "blind_frac": blind / len(records),
        "ego_unmeasured_frac": unmeasured / len(records),
    }


def evaluate_negative(records: list[dict], fps: float, warmup: int) -> dict:
    """Score a clip that contains no airborne target.

    Every confirmed track here is a false alarm by construction. The headline
    number is false tracks per minute, because that is what an operator
    actually experiences - a system that raises a spurious contact every few
    seconds gets ignored, and then it does not matter how good its recall is.
    """
    scored = [r for r in records if int(r["frame"]) >= warmup]
    if not scored:
        return {}

    false_ids: set[int] = set()
    frames_with_lock = 0
    concurrent: list[int] = []
    detections = 0
    for record in scored:
        conf = _confirmed(record)
        for t in conf:
            false_ids.add(int(t["id"]))
        if conf:
            frames_with_lock += 1
        concurrent.append(len(conf))
        detections += len(record.get("detections", []))

    minutes = len(scored) / max(fps, 1e-6) / 60.0
    return {
        "mode": "negative (no target present)",
        "frames_scored": len(scored),
        "duration_min": minutes,
        "false_tracks": len(false_ids),
        "false_tracks_per_min": len(false_ids) / max(minutes, 1e-6),
        "false_lock_rate": frames_with_lock / len(scored),
        "peak_concurrent_false": max(concurrent) if concurrent else 0,
        "mean_concurrent_false": float(np.mean(concurrent)) if concurrent else 0.0,
        "detections_per_frame": detections / len(scored),
    }


def evaluate_sparse(
    records: list[dict],
    gt_path: str,
    fps: float,
    warmup: int,
    radius_factor: float,
    min_radius: float,
) -> dict:
    """Score only the frames that carry a ground-truth label."""
    with open(gt_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    gt_by_frame = {
        int(f["frame"]): f["objects"] for f in meta["frames_data"] if f["objects"]
    }

    evaluated = matched = 0
    total_obs = matched_obs = 0
    ious: list[float] = []
    covering: list[int | None] = []
    first_lock: int | None = None
    first_gt: int | None = None

    for record in records:
        frame = int(record["frame"])
        objects = gt_by_frame.get(frame)
        if objects is None:
            continue  # unlabelled: neither rewarded nor penalised
        if first_gt is None:
            first_gt = frame

        conf = _confirmed(record)
        target = objects[0]
        gc = _center(target["bbox"])
        radius = max(min_radius, radius_factor * float(target.get("size_px", 20)))

        # Precision is only counted on labelled frames, for the same reason the
        # recall side is: a track in an unlabelled frame may well be real.
        for track in conf:
            total_obs += 1
            tc = _center(track["bbox"])
            if np.hypot(tc[0] - gc[0], tc[1] - gc[1]) <= radius:
                matched_obs += 1

        if frame < warmup:
            continue
        evaluated += 1

        best_id, best_iou, best_dist = None, 0.0, float("inf")
        for track in conf:
            tc = _center(track["bbox"])
            dist = float(np.hypot(tc[0] - gc[0], tc[1] - gc[1]))
            if dist <= radius and dist < best_dist:
                best_dist, best_id = dist, int(track["id"])
                best_iou = _iou(track["bbox"], target["bbox"])

        if best_id is not None:
            matched += 1
            ious.append(best_iou)
            if first_lock is None:
                first_lock = frame
        covering.append(best_id)

    switches = 0
    previous = None
    for tid in covering:
        if tid is None:
            continue
        if previous is not None and tid != previous:
            switches += 1
        previous = tid

    recall = matched / evaluated if evaluated else 0.0
    precision = matched_obs / total_obs if total_obs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "mode": "sparse ground truth",
        "frames_scored": evaluated,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "id_switches": switches,
        "latency_frames": (first_lock - first_gt) if (first_lock is not None and first_gt is not None) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--expect-none", action="store_true",
                    help="clip contains no airborne target; score false alarms")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=35,
                    help="frames to skip while the background model settles")
    ap.add_argument("--radius-factor", type=float, default=1.5)
    ap.add_argument("--min-radius", type=float, default=25.0)
    args = ap.parse_args()

    if not args.expect_none and not args.ground_truth:
        ap.error("pass --ground-truth FILE or --expect-none")

    records = load_jsonl(args.tracks)
    if args.expect_none:
        result = evaluate_negative(records, args.fps, args.warmup)
    else:
        result = evaluate_sparse(
            records, args.ground_truth, args.fps, args.warmup,
            args.radius_factor, args.min_radius,
        )
    result.update(mode_stability(records))

    width = max(len(k) for k in result)
    print("=" * 58)
    for key, value in result.items():
        if isinstance(value, float):
            if key.endswith("rate") or key in ("recall", "precision", "f1"):
                print(f"  {key:<{width}} : {value * 100:6.1f} %")
            else:
                print(f"  {key:<{width}} : {value:8.2f}")
        else:
            print(f"  {key:<{width}} : {value}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
